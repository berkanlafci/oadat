OADAT
=======================================================

OADAT: Experimental and Synthetic Clinical Optoacoustic Data for Standardized Image Processing

Paper: Link to paper will be made available after publication.

Dataset: Link to dataset will be made available after publication.

Documentation Website:  

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

Clinical Semi Circle (256 transducer elements), Sparse 128, 64 and 32

**Multisegment**  

Clinical Mutisegment (256 transducer elements) and Linear (128 Elements)

Simulated Mutisegment (256 transducer elements) and Linear (128 Elements)

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

If you use this data in your research, please cite the following paper:

Lafci, B., Ozdemir, F., Dean-Ben, X.L., Razansky, D., Perez-Cruz, F., "OADAT: Experimental and Synthetic Clinical Optoacoustic Data for Standardized Image Processing", (2022).


License
-------------------------------------------------------
The dataset is licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC).  
pyoat package, oadat-evaluate, and oa-armsim projects are licensed under the MIT License.
