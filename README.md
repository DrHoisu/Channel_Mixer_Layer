https://github.com/DrHoisu/Channel_Mixer_Layer
<p align="center" width="100%">
  <img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/logo.png?raw=true' width="90%">
</p>

# Channel Mixer Layer: Multimodal Fusion Towards Machine Reasoning for Spatiotemporal Predictive Learning of Ionospheric Total Electron Content

<p align="left">
<a href="https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
<a href="https://github.com/DrHoisu/Channel_Mixer_Layer/issues" alt="docs">
    <img src="https://img.shields.io/github/issues-raw/DrHoisu/Channel_Mixer_Layer?color=%23FF9600" /></a>
<a href="https://img.shields.io/github/stars/DrHoisu/Channel_Mixer_Layer" alt="arXiv">
    <img src="https://img.shields.io/github/stars/DrHoisu/Channel_Mixer_Layer" /></a>
</p>

[📘Dataset](https://doi.org/10.5281/zenodo.13165939)  |  [🚀Model Zoo](https://doi.org/10.5281/zenodo.13349678)

## Introduction

This repository is the official implementation of our journal paper with the same title. Channel Mixer Layer is a multimodal fusion framework for spatiotemporal predictive learning, which aims to improve the prediction accuracy of predictive channel by inputting auxiliary channel. The program deployment is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL/tree/OpenSTLv0.3.0), a framework of graphic prediction without auxiliary channel input.  We improve it by adding the machine reasoning capability. Note that OpenSTL used in this program is [PyTorch](https://github.com/chengtan9907/OpenSTL/tree/OpenSTLv0.3.0) version instead of the newest Pytorch-Lightning version due to the fixed learning rate [bug](https://github.com/chengtan9907/OpenSTL/issues/113) when this research began. Currently, Channel Mixer Layer only support our [iono_electron](https://doi.org/10.5281/zenodo.13165939) dataset because most datasets for video prediction do **not** have auxiliary data input.

<p align="center" width="100%">
  <img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/Figure1.png?raw=true' width="100%">
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
<p align="center" width="50%">
Table 1 The quantitative performance comparison for the different network backbones, multimodal fusion methods and input channel number $C_{\mathrm{in}}$. The computational complexity is evaluated by parameter number (Param.), Floating-point Operations Per Second (Flops) and inference speed. The prediction accuracy is evaluated by mean squared error (MSE) and mean absolute error (MAE) during low/high solar activity (LSA/HSA) periods.
</p>
<p align="center" width="100%">
  <img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/Table1.png?raw=true' width="100%">
</p>

<p align="right">(<a href="#top">back to top</a>)</p>

#### Overview of Model Zoo

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
our Channel Mixer Layer is deployed as shown by the following process:
<p align="center" width="100%">
  <img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/Figure2.png?raw=true' width="100%">
</p>

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

### static comparison:

<p align="center" width="100%">
  <img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/Figure3.png?raw=true' width="100%">
</p>

### dynamic comparison:

<div align="center">
Ground Truth
<div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/convlstm/1channel/iono_1channel_125_true.gif?raw=true' height="auto" width="260" ></div>

| ConvLSTM Result | Mixer-ConvLSTM Result | 
| :---: | :---: |
| <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/convlstm/1channel/iono_1channel_125_pred.gif?raw=true' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/convlstm/mix_16channel/iono_mix_16channel_125_pred.gif?raw=true' height="auto" width="260" ></div> |

| MIM Result | Mixer-MIM Result |
| :---: | :---: |
|  <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/mim/1channel/iono_1channel_125_pred.gif?raw=true' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/mim/concatenate_16channel/iono_concatenate_16channel_125_pred.gif?raw=true' height="auto" width="260" ></div> |

| PredRNN Result | Mixer-PredRNN Result |
| :---: | :---: |
| <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/predrnn/1channel/iono_1channel_125_pred.gif?raw=true' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/predrnn/mix_16channel/iono_mix_16channel_125_pred.gif?raw=true' height="auto" width="260" ></div> |

| SimVP Result | Mixer-SimVP Result |
| :---: | :---: |
| <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/simvp/1channel/iono_1channel_125_pred.gif?raw=true' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/simvp/concatenate_16channel/iono_concatenate_16channel_125_pred.gif?raw=true' height="auto" width="260" ></div> |

| TAU Result | Mixer-TAU Result |
| :---: | :---: |
| <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/tau/1channel/iono_1channel_125_pred.gif?raw=true' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/tau/concatenate_16channel/iono_concatenate_16channel_125_pred.gif?raw=true' height="auto" width="260" ></div> |

| PredRNN++ Result | Mixer-PredRNN++ Result |
| :---: | :---: |
| <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/predrnnpp/1channel/iono_1channel_125_pred.gif?raw=true' height="auto" width="260" ></div> | <div align=center><img src='https://github.com/DrHoisu/Channel_Mixer_Layer/blob/main/figure/predrnnpp/mix_16channel/iono_mix_16channel_125_pred.gif?raw=true' height="auto" width="260" ></div> |

</div>

## License

This project is released under the [Apache 2.0 license](LICENSE).

<p align="right">(<a href="#top">back to top</a>)</p>
