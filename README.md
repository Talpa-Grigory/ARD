## [CVPR 2025 Oral] Autoregressive Distillation of Diffusion Transformers <br><sub><sub> This repository is a re-implementation of the original work (ARD), recreated based on the author's memory. </sub></sub>
**[Yeongmin Kim](https://sites.google.com/view/yeongmin-space), Sotiris Anagnostidis, Yuming Du, Edgar Schoenfeld, Jonas Kohler, Markos Georgopoulos, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu**  

## Overview
<i>We propose AutoRegressive Distillation (ARD), a method that leverages the historical trajectory of diffusion ODEs to mitigate exposure bias and improve efficiency, achieving strong performance on ImageNet and text-to-image synthesis with significantly fewer steps and minimal computational overhead.</i>
![Teaser image](./assets/figure1.JPG)

## Training

We provide the training bash file train_dbae.sh with dbae_train.py.
Set variables `DATASET_NAME` and `SCHEDULE_TYPE`:
- `DATASET_NAME` sets the dataset. We support FFHQ, CelebA, CelebA-HQ, and LSUN.

We use 4 $\times$ L40S GPUs for FFHQ and LSUN datasets and 4 $\times$ RTX 3090 GPUs for celeba64.

To train, run

```
bash train_dbae.sh $DATASET_NAME $SCHEDULE_TYPE $STO
```

## Dependencies
The requirements for this code are the same as [DiT](https://github.com/facebookresearch/DiT).
