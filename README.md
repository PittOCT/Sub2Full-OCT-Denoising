# Sub2Full-OCT-Denoising
The official code of the paper [Sub2Full: split spectrum to boost OCT despeckling without clean data](https://opg.optica.org/ol/abstract.cfm?URI=ol-49-11-3062).  
## Test
To test the performance of Sub2Full, you can use the instructions as follows:  
- Install necessary python packages using requirements.txt.
- Run test.py
## Citation
If you find this project useful, we would be grateful if you cite our paper：
```
@article{Wang:24,
author = {Lingyun Wang and Jose A Sahel and Shaohua Pi},
journal = {Opt. Lett.},
number = {11},
pages = {3062--3065},
publisher = {Optica Publishing Group},
title = {Sub2Full: split spectrum to boost optical coherence tomography despeckling without clean data},
volume = {49},
month = {Jun},
year = {2024},
url = {https://opg.optica.org/ol/abstract.cfm?URI=ol-49-11-3062},
doi = {10.1364/OL.518906},
abstract = {Optical coherence tomography (OCT) suffers from speckle noise, causing the deterioration of image quality, especially in high-resolution modalities such as visible light OCT (vis-OCT). Here, we proposed an innovative self-supervised strategy called Sub2Full (S2F) for OCT despeckling without clean data. This approach works by acquiring two repeated B-scans, splitting the spectrum of the first repeat as a low-resolution input, and utilizing the full spectrum of the second repeat as the high-resolution target. The proposed method was validated on vis-OCT retinal images visualizing sublaminar structures in the outer retina and demonstrated superior performance over state-of-the-art Noise2Noise (N2N) and Noise2Void (N2V) schemes.},
}
```
