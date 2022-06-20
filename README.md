OADAT
=======================================================

OADAT: Experimental and Synthetic Clinical Optoacoustic Data for Standardized Image Processing

Paper: Link to paper will be made available after publication.

Dataset: Link to dataset will be made available after publication.

Documentation Website: Will be public soon.

Features
-------------------------------------------------------
- Experimental Data
- Simulated Data
- Benchmarks
- Reconstruction Algorithms
- Open Source

Dataset
-------------------------------------------------------

**Experimental Data**  

Experimental data contains raw signals with different transducer arrays and reconstructed OA images with different settings.

**Simulated Data**  

Scripts used to generate simulated acoustic pressure maps are available here: [OA Armsim](https://renkulab.io/gitlab/firat.ozdemir/oa-armsim)

<img src="https://github.com/berkanlafci/oadat/blob/main/docs/images/simulatedDataStatistics_v1.png" width="1000" height="250">
ms = multisegment ||
vc, ss128 = virtual circle, sparse 128 ||
vc, ss64 = virtual circle, sparse 64 ||
vc, ss32 = virtual circle, sparse 32 ||
vc, lv128 = virtual circle, limited view 128 ||
linear = linear part of multisegment array

Transducer Arrays
-------------------------------------------------------

Positions of all array elements are included in the oadat, under arrays subfolder.

**Semi Circle**  

The clinical images acquired with semi circle array are reconstructed using full semi circle (256 transducer elements), sparse 128 (uniform sparsity), sparse 64 (uniform sparsity), sparse 32 (uniform sparsity) and limited view 128 (90 degrees angular coverage).

**Multisegment**  

The clinical images acquired with multisegment array are reconstructed using full multisegment (256 transducer elements) and linear part (128 transducer elements).

The synthetic images simulated with multisegment array are reconstructed using full multisegment (256 transducer elements) and linear part (128 transducer elements).

**Virtual Circle**  

Full Sampling (1024 transducer elements), Sparse 128, 64, 32 and Limited 128

<img src="https://github.com/berkanlafci/oadat/blob/main/docs/images/virtualRingImages_v1.png" width="1000" height="180">

Benchmarks
-------------------------------------------------------

The network architectures and trained weights for benchmarking can be found here: [OADAT Evaluate](https://renkulab.io/gitlab/firat.ozdemir/oadat-evaluate)

Reconstruction Algorithms
-------------------------------------------------------

Image reconstruction algorithms used in this project are included in the following package: [pyoat](https://github.com/berkanlafci/pyoat)

Citation
-------------------------------------------------------

Please cite to this work using the following Bibtex entry:
```
@article{lafci2022oadat,
  doi = {10.48550/ARXIV.2206.08612},
  url = {https://arxiv.org/abs/2206.08612},
  author = {Lafci, Berkan and Ozdemir, Firat and Deán-Ben, Xosé Luís and Razansky, Daniel and Perez-Cruz, Fernando},
  title = {{OADAT}: Experimental and Synthetic Clinical Optoacoustic Data for Standardized Image Processing},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International},
  journal={arXiv preprint arXiv:2206.08612},
}
```

License
-------------------------------------------------------
The dataset is licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC).  
pyoat package, oadat-evaluate, and oa-armsim projects are licensed under the MIT License.
