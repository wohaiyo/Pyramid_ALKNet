# Pyramid ALKNet for Semantic Parsing of Building Facade Image
This project contains the Tensorflow implementation for the proposed Pyramid ALKNet: [[IEEE Xplore]](https://ieeexplore.ieee.org/document/9097171).

### Introduction
As a pixel-level prediction task, semantic segmentation needs large computational cost with enormous parameters to obtain high performance. Recently, due to the increasing demand for autonomous systems and robots, it is significant to make a tradeoff between accuracy and inference speed. In this paper, we propose a novel Depthwise Asymmetric Bottleneck (DAB) module to address this dilemma, which efficiently adopts depth-wise asymmetric convolution and dilated convolution to build a bottleneck structure. Based on the DAB module, we design a Depth-wise Asymmetric Bottleneck Network (DABNet) especially for real-time semantic segmentation, which creates sufficient receptive field and densely utilizes the contextual information. Experiments on Cityscapes and CamVid datasets demonstrate that the proposed DABNet achieves a balance between speed and precision. Specifically, without any pretrained model and postprocessing, it achieves 70.1% Mean IoU on the Cityscapes test dataset with only 0.76 million parameters and a speed of 104 FPS on a single GTX 1080Ti card.
<p align="center"><img width="80%" src="data/framework.png" /></p>

### Installation
- Env: Python 3.6; TensorFlow 0.5-1.5; CUDA 9.0; cuDNN
- Install some packages
```
pip install opencv-python pillow numpy matplotlib imageio .....
```
- Clone this repository
```
git clone https://github.com/wohaiyo/Pyramid_ALKNet
cd Pyramid_ALKNet
```
- One GPU with 11GB is needed

### Dataset
You need to download the two dataset——ECP and RueMonge2014, and put the files in the `data` folder with following structure.
```
├── ecp
|    |── ecp1
|    |   ├── train
|    |   |   ├── img
|    |   |   └── label
|    |   ├── val
|    |   |   ├── img
|    |   |   └── label
|    |   ├── train.txt
|    |   └── val.txt
|    |── ecp2
|    |   ├── train
|    |   |   ├── img
|    |   |   └── label
|    |   ├── val
|    |   |   ├── img
|    |   |   └── label
|    |   ├── train.txt
|    |   └── val.txt
|    |── ecp3
|    |   ├── train
|    |   |   ├── img
|    |   |   └── label
|    |   ├── val
|    |   |   ├── img
|    |   |   └── label
|    |   ├── train.txt
|    |   └── val.txt
|    |── ecp4
|    |   ├── train
|    |   |   ├── img
|    |   |   └── label
|    |   ├── val
|    |   |   ├── img
|    |   |   └── label
|    |   ├── train.txt
|    |   └── val.txt
|    |── ecp5
|    |   ├── train
|    |   |   ├── img
|    |   |   └── label
|    |   ├── val
|    |   |   ├── img
|    |   |   └── label
|    |   ├── train.txt
|    |   └── val.txt
├── RueMonge2014
|    ├── train
|    |    ├── img
|    |    └── label
|    ├── val
|    |    ├── img
|    |    └── label
|    ├── train.txt
|    └── val.txt           
```

### Training

- You can run: `python train.py -h` to check the detail of optional arguments.
Basically, in the `config.py`, you can set the dataset, train type, epochs and batch size, etc.
```
python train.py
```
- training on Cityscapes train set


### Testing
- After training, the checkpoint will be saved at `saves` folder, you can use `eval.py` to get the result.
```
python eval.py
```

### Citation

Please consider citing the [Pyramid ALKNet](https://ieeexplore.ieee.org/document/9097171) if it's helpful for your research.
```
@article{ma2020pyramid,
  title={Pyramid ALKNet for Semantic Parsing of Building Facade Image},
  author={Ma, Wenguang and Ma, Wei and Xu, Shibiao and Zha, Hongbin},
  booktitle={IEEE Geoscience and Remote Sensing Letters},
  year={2020},
  publisher={IEEE}
}