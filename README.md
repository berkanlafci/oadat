# OADAT

[![arXiv](https://img.shields.io/badge/Preprint-arXiv-b31b1b)](https://arxiv.org/abs/2206.08612)
[![Documentation](https://img.shields.io/badge/Documentation-OADAT-brightgreen)](https://berkanlafci.github.io/oadat/)
[![Data](https://img.shields.io/badge/Data-Research%20Collection-blue)](https://www.research-collection.ethz.ch/handle/20.500.11850/551512)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)  

OADAT: Experimental and Synthetic Clinical Optoacoustic Data for Standardized Image Processing

Paper: [arXiv Link](https://arxiv.org/abs/2206.08612)  

Dataset: Will be public in [ETH Zurich Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/551512) upon paper acceptance.  

Documentation: [Website Link](https://berkanlafci.github.io/oadat/)

## Features

- **Experimental Data:** OADAT provides experimental clinical data of forearm raw signals and reconstructed images.
- **Simulated Data:** The paper provides methods to simulate forearm acoustic pressure maps, raw signals and reconstructed images using optoacoustic forward model and backprojection reconstruction algorithm ([OA Armsim](https://renkulab.io/gitlab/firat.ozdemir/oa-armsim)). 
- **Benchmarks:** 18 benchmark experiments are conducted for 3 different tasks, namely sparse acquisition, limited view and segmentation ([OADAT Evaluate](https://renkulab.io/gitlab/firat.ozdemir/oadat-evaluate)).   
- **Reconstruction Algorithm:** We provide image reconstruction algorithm called backprojection to generate optoacoustic images from raw signals ([pyoat](https://github.com/berkanlafci/pyoat)).  
- **Open Source:** All simulation, benchmarking and reconstruction algorithms are presented publicly. We provide easy-to-use scripts for data reading, simulations, reconstructions and benchmarks.

## Dataset

### Experimental Data

Experimental data contains raw signals with different transducer arrays and reconstructed OA images with different settings.

<img src="https://github.com/berkanlafci/oadat/blob/main/docs/images/semiCircleImages_v1.png" width="1000" height="180">

### Simulated Data

Scripts used to generate simulated acoustic pressure maps are available here: [OA Armsim](https://renkulab.io/gitlab/firat.ozdemir/oa-armsim)

Simulated virtual circle array images are shown in the figure below with different reconstruction schemes.
- Full Sampling (1024 transducer elements)
- Sparse 128
- Sparse 64
- Sparse 32
- Limited 128

<img src="https://github.com/berkanlafci/oadat/blob/main/docs/images/virtualRingImages_v1.png" width="1000" height="180">

Distribution of features in simulated data are summarized in the figure below. The plots give information about:  
- number of pixels (size) distribution for each vessel,  
- number of vessels distribution in each image,  
- PSNR values compared to full sampling virtual circle reconstructions (artifact free images).

<img src="https://github.com/berkanlafci/oadat/blob/main/docs/images/simulatedDataStatistics_v1.png" width="1000" height="250">
ms = multisegment ||
vc, ss128 = virtual circle, sparse 128 ||
vc, ss64 = virtual circle, sparse 64 ||
vc, ss32 = virtual circle, sparse 32 ||
vc, lv128 = virtual circle, limited view 128 ||
linear = linear part of multisegment array

## Transducer Arrays

Positions of all array elements are included in the oadat, under arrays subfolder.

### Semi Circle

The clinical images acquired with semi circle array are reconstructed using full semi circle (256 transducer elements), sparse 128 (uniform sparsity), sparse 64 (uniform sparsity), sparse 32 (uniform sparsity) and limited view 128 (90 degrees angular coverage).

### Multisegment

The clinical images acquired with multisegment array are reconstructed using full multisegment (256 transducer elements) and linear part (128 transducer elements).

The synthetic images simulated with multisegment array are reconstructed using full multisegment (256 transducer elements) and linear part (128 transducer elements).

### Virtual Circle

The array is generated to simulate images with 360 degree angular coverage which results in artifact free reconstructions. It contains 1024 transducer elements distributed over a full circle with equal distance. The radius of the transducer array is kept equal to semi circle array (40 mm) to allow comparison between simulations and experimental acquisitions.

## Benchmarks

The network architectures and trained weights for benchmarking can be found here: [OADAT Evaluate](https://renkulab.io/gitlab/firat.ozdemir/oadat-evaluate)

We define 18 experiments based on 3 tasks (sparse reconstructions, limited view corrections and segmentation). Sparse sampling and limited view corrections are grouped under image translation task.

### Image Translation

Through a list of permutations of our datasets, we can define several pairs of varying difficulty of image translation experiments where target images are also available.

- **Sparse Reconstructions:** We present sparse reconstructions of both single wavelength forearm dataset (SWFD) and simulated cylinders dataset (SCD) for semi circle and virtual ring arrays. Accordingly, sparse sampling correction experiments learn mapping functions where the task is correction of sparse sampling (ss) artifacts from the reduced number of elements used for image reconstruction with different array geometries.

- **Limited View:** Further, we offer limited view reconstructions for all datasets and for all arrays. Accordingly, limited view correction experiments learn mapping functions where the task is correction of limited view (lv) artifacts from the reduced angular coverage used for image reconstruction with different array geometries.

### Segmentation

Simulated cylinders dataset (SCD) includes pixel annotations for skin curve, vessels and background. In addition to segmentation of these structures on the ideal reconstructions, we define segmentation task on sparse sampling and limited view reconstructions that contain the relevant artifacts encountered in experimental data. All data in segmentation task is generated from SCD and the objective is to match the ground truth annotations of the acoustic pressure map.


## Reconstruction Algorithms

Image reconstruction algorithms used in this project are included in the following package: [pyoat](https://github.com/berkanlafci/pyoat)  

We use backprojection algorithm in this study to generate OA images from the acquired signals. The algorithm is based on delay and sum beamforming approach. First, a mesh grid is created for desired field of view. Then, the distance between the points of the mesh grid and transducer elements are calculated based on the known locations of the array elements and the pixels. Time of flight (TOF) is obtained through dividing the distance by the SoS values that are assigned based on the temperature of the imaging medium and tissue properties. The corresponding signals are picked from signal matrix based on TOF. Then, the signals are added up to create images. The clinical and simulated data are reconstructed with SoS of 1510 m/s in this study as the simulations and the experiments were done at the corresponding imaging medium temperature.

## Citation

Please cite to this work using the following Bibtex entry:
```
@article{lafci2022oadat,
  author      = {Lafci, Berkan and 
                Ozdemir, Firat and 
                Deán-Ben, Xosé Luís and 
                Razansky, Daniel and 
                Perez-Cruz, Fernando},
  title       = {OADAT: Experimental and Synthetic Clinical Optoacoustic Data for Standardized Image Processing},
  publisher   = {arXiv},
  year        = {2022},
  copyright   = {Creative Commons Attribution Non Commercial Share Alike 4.0 International},
  journal     = {arXiv preprint arXiv:2206.08612},
  doi         = {10.48550/ARXIV.2206.08612},
  url         = {https://arxiv.org/abs/2206.08612}
}
```

## License

The dataset is licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC).  
pyoat package, oadat-evaluate, and oa-armsim projects are licensed under the MIT License.
