# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import logging
import os
import pickle
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import blobfile as bf
import numpy as np
import torch.distributed as dist

from dataset import load_data
from diffusers.models import AutoencoderKL

from diffusion import create_diffusion
from download import find_model
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in [
            "npz"
        ]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        print(i)
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    all_files = _list_image_files_recursively(args.data_path) ## this take some time at initial
    # with open('all_files.pickle', 'wb') as f: ## use it after first run
    #     pickle.dump(all_files, f)

    # Setup DDP:
    dist.init_process_group("nccl")
    assert (
        args.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    # Setup data:

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace(
        "/", "-"
    )  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    data = load_data(
        all_files=all_files,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        image_size=args.image_size,
        class_cond=True,
        random_crop=False,
        random_flip=False,
        predx0=args.predx0,
        all=True,
    )

    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda pil_image: center_crop_arr(pil_image, args.image_size)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    dataset = ImageFolder(args.real_data_path, transform=transform)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create model:
    assert (
        args.image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    from models_ARD import DiT_models
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Initilize from pre-trained diffusion teacher
    if args.ckpt:
        print(f"Loading teacher from {args.ckpt_path}")
        if args.ckpt_path == None:
            ckpt_path = f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
            state_dict = find_model(ckpt_path)
            model.load_state_dict(state_dict, strict=False)
            model.copy_weights()
            ema = deepcopy(model).to(
                device
            )  # Create an EMA of the model for use after training
        else:
            ckpt_path = args.ckpt_path
            state_dict = find_model(ckpt_path)
            model.load_state_dict(state_dict["model"], strict=True)
            model = model.to(device)
            ema = deepcopy(model)  # Create an EMA of the model for use after training
            ema.load_state_dict(state_dict["ema"])
            opt.load_state_dict(state_dict["opt"])
            ema = ema.to(device)
    else:
        ema = deepcopy(model).to(device)
    requires_grad(ema, False)


    from models_discriminator import DiT_models, OneDiscriminator

    Embedder = DiT_models[args.model](
        input_size=latent_size, num_classes=args.num_classes
    )
    ckpt_path = f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    Embedder.load_state_dict(state_dict, strict=False) ## embedder is teacher checkpoint
    Embedder = DDP(Embedder.to(device), device_ids=[rank])
    Embedder.eval()

    # 1. Discriminator Heads
    discriminators = [OneDiscriminator() for _ in range(28)]

    # 2. Optimizer
    opt_dis = torch.optim.AdamW(
        [param for dis in discriminators for param in dis.parameters()],
        lr=args.dis_lr,
        weight_decay=0,
    )

    # 3. DDP wrapping
    discriminators = [
        DDP(dis.to(device), device_ids=[rank], find_unused_parameters=True)
        for dis in discriminators
    ]


    # Note that parameter initialization is done within the DiT constructor
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(
        timestep_respacing=""
    )  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare models for training:
    update_ema(
        ema, model.module, decay=0
    )  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    gan_gen_loss = 0
    gan_dis_loss = 0

    start_time = time()

    for epoch in range(5000):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            ## Real
            real_x = x.to(device)
            real_y = y.to(device)
            with torch.no_grad():
                real_x = vae.encode(real_x).latent_dist.sample().mul_(0.18215)

            ## Fake
            z0, z5, z10, z15, _, zT, y = next(data)  # (batch, 4, 32, 32), (batch, 4, 32, 32), (batch)
            z0 = z0.to(device)
            z5 = z5.to(device)
            z10 = z10.to(device)
            z15 = z15.to(device)
            zT = zT.to(device)
            y = y.to(device)

            t = torch.ones((z0.shape[0]), device=device) * 999.0
            model_kwargs = dict(y=y)

            ## Generator
            for dis in discriminators:
                dis.eval()


            loss_dict, pred_z0 = diffusion.AR_4steps_losses_oneshot_gan_all_hinge(
                model,
                z0,
                z5,
                z10,
                z15,
                zT,
                t,
                args.stack,
                model_kwargs,
                Embedder,
                discriminators
                )


            ## Calculate Adapteive Weights
            g_grads = torch.autograd.grad(
                loss_dict["gan"].mean(),
                model.module.final_layer4.parameters(),
                retain_graph=True,
                allow_unused=True,
            )[0]
            nll_grads = torch.autograd.grad(
                loss_dict["loss"].mean(),
                model.module.final_layer4.parameters(),
                retain_graph=True,
                allow_unused=True,
            )[0]


            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
            if train_steps < 3000:
                d_weight = d_weight * 0.0
            else:
                d_weight = d_weight * args.ld

            ## Update Generator
            loss = loss_dict["loss"].mean() + d_weight * loss_dict["gan"].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
            update_ema(ema, model.module)

            for dis in discriminators:
                dis.train()


            loss_dict_dis = diffusion.dis_all_hinge(
                pred_z0,
                model_kwargs,
                real_x,
                real_y,
                discriminators,
                t,
                Embedder,
            )
            loss_dis = loss_dict_dis["gan_dis"].mean()
            opt_dis.zero_grad()
            loss_dis.backward()
            opt_dis.step()

            # Log loss values every 100 steps
            running_loss += loss_dict["loss"].mean().item()
            gan_gen_loss += loss_dict["gan"].mean().item()
            gan_dis_loss += loss_dict_dis["gan_dis"].mean().item()

            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_gan_gen_loss = torch.tensor(gan_gen_loss / log_steps, device=device)
                avg_gan_dis_loss = torch.tensor(gan_dis_loss / log_steps, device=device)

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f} Gan_Gen_Loss: {avg_gan_gen_loss:.4f} Gan_Dis_Loss: {avg_gan_dis_loss:.4f} Train Steps/Sec: {steps_per_sec:.2f}"
                )

                # Reset monitoring variables:
                running_loss = 0
                gan_gen_loss = 0
                log_steps = 0
                gan_dis_loss = 0
                start_time = time()

            # Save checkpoint every 10k
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()
    model.eval()  # important! This disables randomized embedding dropout
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":

    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path",type=str,default="/data/home/yeongminkim/Codes/DiT/samples_trajectory/latent_trajectory/zt/0")
    parser.add_argument("--real-data-path", type=str,default="/path/to/imagenet/train")
    parser.add_argument("--predx0", type=int, default=1)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=40_000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)

    parser.add_argument("--ckpt", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="/{Path}/results/checkpoints/0300000.pt") ## Set it pre-trained ckpt with regression loss
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dis_lr", type=float, default=1e-3)
    parser.add_argument("--stack", type=int, default=6)
    parser.add_argument("--ld", type=float, default=10.0)
    parser.add_argument("--hinge", type=bool, default=False)
    parser.add_argument("--newnew", type=bool, default=False)
    parser.add_argument("--grad_clip", type=float, default=1.0) ## try 0.7 or 0.4 if unstable

    args = parser.parse_args()
    main(args)
