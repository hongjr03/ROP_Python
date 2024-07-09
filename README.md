# An Unofficial Python Implementation of Rank-One Prior (CVPR21 & TPAMI23)

This repository contains an unofficial Python implementation of the Rank-One Prior method. The original paper is available [here](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Rank-One_Prior_Toward_Real-Time_Scene_Recovery_CVPR_2021_paper.pdf) and [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9969127).

The file tree is as follows:

```plaintext
.
│  demo_Rank1.m
│  demo_Rank1plus.m
│  psf2otf.py
│  README.md
│  ROP+.py
│  ROP.py
│
└─images
        hazy1.png
        sandstorm1.png
        underw1.jpg
```

The `ROP.py` and `ROP+.py` files contain the Rank-One Prior and Rank-One Prior Plus methods, respectively, which are implemented by me based on the original MATLAB code (`demo_Rank1.m` and `demo_Rank1plus.m`) provided by the authors.

The `psf2otf.py` file contains a function that converts a point spread function (PSF) to an optical transfer function (OTF), borrowed from [ftvd.py](https://github.com/JoshuaEbenezer/ftvd/blob/master/ftvd.py).

There are still differences on the results between the original MATLAB code and my Python implementation. Looking forward to any suggestions or contributions.
