## [CVPR 2025 Oral] Autoregressive Distillation of Diffusion Transformers <br><sub><sub> This repository provides a re-implementation of the original work (ARD), reconstructed from the author's recollection. </sub></sub>
**[Yeongmin Kim](https://sites.google.com/view/yeongmin-space), Sotiris Anagnostidis, Yuming Du, Edgar Schoenfeld, Jonas Kohler, Markos Georgopoulos, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu**  

## Overview
<i>We propose AutoRegressive Distillation (ARD), a method that leverages the historical trajectory of diffusion ODEs to mitigate exposure bias and improve efficiency, achieving strong performance on ImageNet and text-to-image synthesis with significantly fewer steps and minimal computational overhead.</i>
![Teaser image](./assets/figure1.JPG)

## Dependencies
The requirements for this code are the same as [DiT](https://github.com/facebookresearch/DiT).

## Training
Set variables `DATASET_NAME` and `SCHEDULE_TYPE`:
- `DATASET_NAME` sets the dataset. We support FFHQ, CelebA, CelebA-HQ, and LSUN.
We use 4 $\times$ L40S GPUs for FFHQ and LSUN datasets and 4 $\times$ RTX 3090 GPUs for celeba64.
To train, run
```
bash train_dbae.sh $DATASET_NAME $SCHEDULE_TYPE $STO
```

## Fine-tuning with GAN loss
Set variables `DATASET_NAME` and `SCHEDULE_TYPE`:
- `DATASET_NAME` sets the dataset. We support FFHQ, CelebA, CelebA-HQ, and LSUN.
We use 4 $\times$ L40S GPUs for FFHQ and LSUN datasets and 4 $\times$ RTX 3090 GPUs for celeba64.
To train, run
```
bash train_dbae.sh $DATASET_NAME $SCHEDULE_TYPE $STO
```

## Generation




## Reference
If you find the code useful for your research, please consider citing
```bib
@inproceedings{
}
```
