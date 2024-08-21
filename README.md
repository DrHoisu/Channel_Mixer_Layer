<p align="center" width="100%">
  <img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/246222783-fdda535f-e132-4fdd-8871-2408cd29a264.png' width="50%">
</p>

# Channel Mixer Layer: Multimodal Fusion Towards Machine Reasoning for Spatiotemporal Predictive Learning of Ionospheric Total Electron Content

<p align="left">
<a href="https://arxiv.org/abs/2306.11249" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2306.11249-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/blob/master/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
<!-- <a href="https://huggingface.co/OpenSTL" alt="Huggingface">
    <img src="https://img.shields.io/badge/huggingface-OpenSTL-blueviolet" /></a> -->
<a href="https://openstl.readthedocs.io/en/latest/" alt="docs">
    <img src="https://readthedocs.org/projects/openstl/badge/?version=latest" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/issues" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/chengtan9907/SimVPv2?color=%23FF9600" /></a>
<a href="https://github.com/chengtan9907/OpenSTL/issues" alt="resolution">
    <img src="https://img.shields.io/badge/issue%20resolution-1%20d-%23B7A800" /></a>
<a href="https://img.shields.io/github/stars/chengtan9907/OpenSTL" alt="arXiv">
    <img src="https://img.shields.io/github/stars/chengtan9907/OpenSTL" /></a>
</p>
[🛠️Installation](docs/en/install.md)  |  [📘Dataset](https://doi.org/10.5281/zenodo.13165939)  |  [🚀Model Zoo](https://doi.org/10.5281/zenodo.13349678)  |  [👀Visualization](docs/en/visualization/video_visualization.md)  |  [👍Citation](docs/en/changelog.md)

## Introduction

This repository is the official implementation of our journal paper with the same title. Channel Mixer Layer is a multimodal fusion framework for spatiotemporal predictive learning, which aims to improve the prediction accuracy of predictive channel by inputting auxiliary channel. The program deployment is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL/tree/OpenSTLv0.3.0), a framework of graphic prediction without auxiliary channel input.  We improve it by adding the machine reasoning capability. Note that OpenSTL used in this program is [PyTorch](https://github.com/chengtan9907/OpenSTL/tree/OpenSTLv0.3.0) version instead of the newest Pytorch-Lightning version due to the fixed learning rate [bug](https://github.com/chengtan9907/OpenSTL/issues/113) when this research began. Currently, Channel Mixer Layer only support our [iono_electron](https://doi.org/10.5281/zenodo.13165939) dataset because most datasets for video prediction do **not** have auxiliary data input.

<p align="center" width="100%">
  <img src='https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/246222226-61e6b8e8-959c-4bb3-a1cd-c994b423de3f.png' width="90%">
</p>

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview
<details open>
<summary>Hardware Recommendation</summary>

- `RAM memory` > 16 GB
- `GPU` > Nvidia RTX 3090 with the VRAM larger than 24 GB

</details>

<details open>
<summary>Code Structures</summary>

- `scripts/` contains experiment execution scripts.
- `openstl/api` contains an experiment runner.
- `openstl/core` contains core training plugins and metrics.
- `openstl/datasets` contains datasets and dataloaders.
- `openstl/methods/` contains training methods for various video prediction methods.
- `openstl/models/` contains the main network architectures of various video prediction methods.
- `openstl/modules/` contains network modules and layers.
- `tools/` contains the executable python files `tools/train.py` and `tools/test.py` with possible arguments for training, validating, and testing pipelines.

</details>

## Installation

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
git clone https://github.com/DrHoisu/Channel_Mixer_Layer
cd Channel_Mixer_Layer
conda env create -f environment.yml
conda activate Mixer
python setup.py develop
```

Please refer to [install.md](docs/en/install.md) for more detailed instructions.

## Iono_Electron Data-set

Download Iono_Electron Data-set from [website](https://doi.org/10.5281/zenodo.13165939) or using the following command:

```shell
bash tools/prepare_data/download_iono.sh
```

## Pretrained Model

Download pretrained models from [website](https://doi.org/10.5281/zenodo.13349678) or using the following command:

```shell
bash tools/prepare_data/download_pretrained_model.sh
```

## Training and Testing

Please see [get_started.md](docs/en/get_started.md) for the basic usage. All training and testing commands are listed under `scripts/` directory Here is an example to evaluate the prediction accuracy of the pretrained models with different input channel number and different multimodal fusion methods.
```shell
bash scripts/iono/convlstm/convlstm.sh
bash scripts/iono/e3dlstm/e3dlstm.sh
bash scripts/iono/mau/mau.sh
bash scripts/iono/mim/mim.sh
bash scripts/iono/predrnn/predrnn.sh
bash scripts/iono/predrnnpp/predrnnpp.sh
bash scripts/iono/predrnnv2/predrnnv2.sh
bash scripts/iono/simvp/simvp.sh
bash scripts/iono/tau/tau.sh
```

<p align="right">(<a href="#top">back to top</a>)</p>

#### Overview of Model Zoo

We support various spatiotemporal prediction methods and provide [benchmarks](https://github.com/chengtan9907/OpenSTL/tree/master/docs/en/model_zoos) on various STL datasets. We are working on add new methods and collecting experiment results.

* Spatiotemporal Prediction Methods.

    <details open>
    <summary>Currently supported methods</summary>

    - [x] [ConvLSTM](https://arxiv.org/abs/1506.04214) (NeurIPS'2015)
    - [x] [PredNet](https://openreview.net/forum?id=B1ewdt9xe) (ICLR'2017)
    - [x] [PredRNN](https://dl.acm.org/doi/abs/10.5555/3294771.3294855) (NeurIPS'2017)
    - [x] [PredRNN++](https://arxiv.org/abs/1804.06300) (ICML'2018)
    - [x] [E3D-LSTM](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2018)
    - [x] [MIM](https://arxiv.org/abs/1811.07490) (CVPR'2019)
    - [x] [CrevNet](https://openreview.net/forum?id=B1lKS2AqtX) (ICLR'2020)
    - [x] [PhyDNet](https://arxiv.org/abs/2003.01460) (CVPR'2020)
    - [x] [MAU](https://openreview.net/forum?id=qwtfY-3ibt7) (NeurIPS'2021)
    - [x] [PredRNN.V2](https://arxiv.org/abs/2103.09504v4) (TPAMI'2022)
    - [x] [SimVP](https://arxiv.org/abs/2206.05099) (CVPR'2022)
    - [x] [SimVP.V2](https://arxiv.org/abs/2211.12509) (ArXiv'2022)
    - [x] [TAU](https://arxiv.org/abs/2206.12126) (CVPR'2023)
    - [x] [DMVFN](https://arxiv.org/abs/2303.09875) (CVPR'2023)

    </details>

You can understand our Channel Mixer Layer more intuitively by the following command:
```shell
python Understand_Mixer.py
```


## Visualization

We present visualization examples of ConvLSTM below. For more detailed information, please refer to the [visualization](docs/en/visualization/).

The visualization of 1-channel graphic prediction with ConvLSTM model for the No.125 global TEC sequence is running by the following command:

```shell
python tools/visualizations/vis_video.py -d iono -w work_dirs/iono/convlstm/1channel --index 125 --save_dirs visualization/iono/convlstm/1channel/
```
The visualization of 16-channel multimodal fusion prediction with Mixer-ConvLSTM model for the No.125 global TEC sequence is running by the following command:

```shell
python tools/visualizations/vis_video.py -d iono -w work_dirs/iono/convlstm/mix_16channel --index 125 --save_dirs visualization/iono/convlstm/mix_16channel/
```

<div align="center">
Ground Truth
<div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_ConvLSTM.gif' height="auto" width="260" ></div>

| ConvLSTM Result | Mixer-ConvLSTM | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/mmnist_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_fashionmnist_ConvLSTM.gif' height="auto" width="260" ></div> |

| Moving MNIST-CIFAR | KittiCaltech |
| :---: | :---: |
|  <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/moving_mnist_cifar_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kitticaltech_ConvLSTM.gif' height="auto" width="260" ></div> |

| KTH | Human 3.6M | 
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/kth20_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-video/human_ConvLSTM.gif' height="auto" width="260" ></div> |

| Traffic - in flow | Traffic - out flow |
| :---: | :---: |
| <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_in_flow_ConvLSTM.gif' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/chengtan9907/OpenSTL/releases/download/vis-traffic/taxibj_out_flow_ConvLSTM.gif' height="auto" width="260" ></div> |

</div>

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you are interested in our repository or our paper, please cite our journal paper:

`Plain Text:`

```
@inproceedings{tan2023openstl,
  title={OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning},
  author={Tan, Cheng and Li, Siyuan and Gao, Zhangyang and Guan, Wenfei and Wang, Zedong and Liu, Zicheng and Wu, Lirong and Li, Stan Z},
  booktitle={Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}

```
`bibtex:`
```
@inproceedings{tan2023openstl,
  title={OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning},
  author={Tan, Cheng and Li, Siyuan and Gao, Zhangyang and Guan, Wenfei and Wang, Zedong and Liu, Zicheng and Wu, Lirong and Li, Stan Z},
  booktitle={Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}

```

<p align="right">(<a href="#top">back to top</a>)</p>
