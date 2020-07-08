
This is a Semantic Segmentation project with FCN-8s architecture. The paper can be found [here as pdf](https://arxiv.org/pdf/1411.4038.pdf).

- Dataset

As a dataset I used KITTI Semantic pixel-level dataset which can be downloaded from [here](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015). The dataset structure is as shown below.

```
├── testing
│   └── image_2
│       ├── *.png
│       ├── ...
└── training
    ├── image_2
    │   ├── *.png
    │   ├── ...
    ├── instance
    │   ├── *.png
    │   ├── ...
    ├── semantic
    │   ├── *.png
    │   ├── ...
    └── semantic_rgb
        ├── *.png
        ├── ...
```
