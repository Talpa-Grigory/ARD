## [CVPR 2025 Oral] Autoregressive Distillation of Diffusion Transformers <br><sub><sub> This repository provides a re-implementation of the original work (ARD), reconstructed from the author's recollection. </sub></sub>
**[Yeongmin Kim](https://sites.google.com/view/yeongmin-space), Sotiris Anagnostidis, Yuming Du, Edgar Schoenfeld, Jonas Kohler, Markos Georgopoulos, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu**  

## Overview
<i>We propose AutoRegressive Distillation (ARD), a method that leverages the historical trajectory of diffusion ODEs to mitigate exposure bias and improve efficiency in distillation. ARD achieves strong performance on ImageNet and text-to-image synthesis with significantly fewer steps and minimal computational overhead.</i>
![Teaser image](./assets/figure1.JPG)

## Dependencies
The requirements for this code are the same as [DiT](https://github.com/facebookresearch/DiT).

## Training
```
bash train.sh
```

## Fine-tuning with GAN loss
```
bash train.sh
```

## Generation
```
bash generate.sh
```

## Performance
We follow the evaluation protocol of [ADM](https://github.com/openai/guided-diffusion/tree/main/evaluations).

## Citation
If you find the code useful for your research, please consider citing
```bib
@inproceedings{
}
```
