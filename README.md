## [CVPR 2025 Oral] Autoregressive Distillation of Diffusion Transformers <br><sub><sub> This repository provides a re-implementation of the original work (ARD), reconstructed from the author's recollection. </sub></sub>
**[Yeongmin Kim](https://sites.google.com/view/yeongmin-space), [Sotiris Anagnostidis](https://sanagnos.pages.dev/), [Yuming Du](https://dulucas.github.io/), [Edgar Schoenfeld](https://edgarschnfld.github.io/), [Jonas Kohler](https://scholar.google.de/citations?user=a1rCLUMAAAAJ&hl=de), [Markos Georgopoulos](https://scholar.google.com/citations?user=id7vw0UAAAAJ&hl=en), [Albert Pumarola](https://www.albertpumarola.com/), [Ali Thabet](https://www.alithabet.com/), [Artsiom Sanakoyeu](https://gdude.de/)**  

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
![Teaser image](./assets/figure2.JPG)
![Teaser image](./assets/figure3.JPG)

## Citation
If you find the code useful for your research, please consider citing
```bib
@inproceedings{
}
```
