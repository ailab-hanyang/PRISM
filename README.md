<div align="center">

# PRISM: Polarimetric Road-surface Intelligent Sensing and Measurement Dataset

<img src="images/ailab-logo.png" height="125" alt="Automotive Intelligence Laboratory">


<p align="center">
  <a href="https://neurips.cc/virtual/2026/loc/paris/poster/138944">
    <img alt="paper" src="https://img.shields.io/badge/Paper-NeurIPS%202026%20E%26D-gray?style=flat&labelColor=555&color=EE4C2C">
  </a>
  <a href="https://huggingface.co/datasets/NeurIPS-2026-PRISM/PRISM-Dataset">
    <img alt="huggingface" src="https://img.shields.io/badge/Hugging%20Face-Dataset-gray?style=flat&logo=huggingface&logoColor=FFD21E&labelColor=gray&color=0078D7">
  </a>
  <a href="LICENSE">
    <img alt="license" src="https://img.shields.io/badge/License-Apache%202.0-gray?style=flat&labelColor=gray&color=green">
  </a>
</p>

<h4 align="center">
  <a href="mailto:minseojung@hanyang.ac.kr">Minseo Jung</a>, <a href="mailto:chanwookim01@hanyang.ac.kr">Chanwoo Kim</a>, <a href="mailto:dongguekim@hanyang.ac.kr">Donggue Kim</a>, <a href="mailto:hyunjunkim@hanyang.ac.kr">Hyunjun Kim</a>, <a href="mailto:jeongwoopark@hanyang.ac.kr">Jeongwoo Park</a>, <a href="mailto:kichunjo@hanyang.ac.kr">Kichun Jo</a>
</h4>
</div>

<br>

This is the official implementation of **"PRISM: Polarimetric Road-surface Intelligent Sensing and Measurement Dataset".**

PRISM is a polarimetric road-surface dataset of 47,098 time-synchronized frames that pair trichromatic linear polarization with co-boresighted RGB, 128-channel LiDAR, and RTK-GNSS/INS, collected on proving grounds and open roads under clear, overcast, rainy, foggy, and snowy conditions. It ships with dense road-surface elevation ground truth, frame-level material and surface-state labels, two benchmark tracks (road-surface condition classification and bird's-eye-view elevation estimation), reference baselines, and evaluation scripts. This repository contains the datasets tooling, training and evaluation code, and pretrained checkpoints for every reported baseline.

---

## News<a name="changelog"></a>
- **`[2026/11/04]`** Nighttime Extension Data is acquired and now available on Hugging Face 🌙
- **`[2026/09/24]`** PRISM is **accepted to the NeurIPS 2026 Evaluations & Datasets Track!** 🎉🎉
- **`[2026/05/06]`** The full dataset and codebase is released

## Table of Contents
1. [Repository Folder Tree](#repository-folder-tree)
2. [Getting started](#getting-started)
3. [Pretrained Checkpoints](#pretrained-checkpoints)
4. [Citation](#citation)
5. [License](#license)
6. [Acknowledgement](#acknowledgement)

## Repository Folder Tree<a name="repository-folder-tree"></a>
```shell
PRISM
│   ├── config
│   │   ├── extrinsics
│   │   │   └── ...                     # Per-scenario extrinsic calibration between polarimetric/RGB cameras and LiDAR
│   │   └── calibration.yaml            # Shared intrinsic parameters and LiDAR-to-ego transform
│   ├── dataset_processor
│   │   ├── config
│   │   │   ├── gt_config.yaml          # Ground truth generation settings
│   │   │   └── pkl_config.yaml         # Pickle file generation settings
│   │   ├── scripts
│   │   │   ├── convert_pkl.py          # Converts raw data into .pkl files for train/test dataloaders
│   │   │   └── create_gt.py            # Generates ground truth labels (road elevation, surface condition, etc.)
│   │   ├── tools
│   │   │   ├── gt_processor.py         # Ground truth generation pipeline
│   │   │   └── visualize_pkl.py        # Visualization tool for inspecting .pkl files
│   ├── project                         # benchmarks
│   │   ├── road-surface-condition-classification   
│   │   │   └── classifier (EfficientNet, MobileNet)
│   │   └── road-surface-elevation-estimation
│   │       └── RoadBEV
│   ├── .gitignore
│   ├── LICENSE                         # Apache 2.0
│   ├── download_dataset.py             # Script for downloading the dataset from Hugging Face
│   ├── requirements.txt
│   └── README.md
```


## Getting started<a name="getting-started"></a>

- [Download PRISM-Dataset](./docs/download.md)
- [Environment Setup](./docs/environment_setup.md)
- [Experiment Setup](./docs/experiment_setup.md)
- [Training](./docs/training.md)
- [Evaluation](./docs/evaluation.md)


## Pretrained Checkpoints<a name="pretrained-checkpoints"></a>

We provide pretrained checkpoints for all benchmark models across 6 input modality variants.

**Input variants:**
| Variant | Description |
|---------|-------------|
| `rgb` | RGB-camera RGB |
| `polar-rgb` | Pol-camera RGB (demosaiced) |
| `mono-polar` | DoLP $+$ $(\sin 2\phi, \cos 2\phi)$ AoLP |
| `tri-polar` | per-ch DoLP $+$ per-ch $(\sin 2\phi, \cos 2\phi)$ AoLP |
| `rgb-mono` | Pol-camera RGB $\mid$ DoLP $+$ $(\sin 2\phi, \cos 2\phi)$ AoLP |
| `rgb-tri` | Pol-camera RGB $\mid$ per-ch DoLP $+$ per-ch $(\sin 2\phi, \cos 2\phi)$ AoLP |

### Road Surface Condition Classification

| Model | `rgb` | `polar-rgb` | `mono-polar` | `tri-polar` | `rgb-mono` | `rgb-tri` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| EfficientNet-B6 | [link](https://drive.google.com/file/d/1bLos1jUqLo2p1W2_l45VNXc2TgZquJuO/view?usp=drive_link) | [link](https://drive.google.com/file/d/1RS-BhfZZtAtGAxsMOsdhIV_Pyvo4wUP0/view?usp=drive_link) | [link](https://drive.google.com/file/d/1Lai61i6mOILHoMcolDP3Dz6aXSlamEUM/view?usp=drive_link) | [link](https://drive.google.com/file/d/1dudPGQvvlMRmn5fXr0DlPDOrnN_Z46pw/view?usp=drive_link) | [link](https://drive.google.com/file/d/1OWcsUV1yNIOK-lpkcW1wPAkkULUza9ML/view?usp=drive_link) | [link](https://drive.google.com/file/d/18g3PwYxW0PBe0dM61qkqHUfQ52R0pLY2/view?usp=drive_link) |
| MobileNet-V3-Large | [link](https://drive.google.com/file/d/1pgTjS4-D-jWeZ2I_Ppu4GqFx47WnrPlN/view?usp=drive_link) | [link](https://drive.google.com/file/d/1YqqlACV6_93Z-pvtBo-7ubJ_hJ3UnoUn/view?usp=drive_link) | [link](https://drive.google.com/file/d/1QUYxvxNysmVtPu7opHUHYL2NLyyIpYmg/view?usp=drive_link) | [link](https://drive.google.com/file/d/1V0b0RB0nISy32I9E_odO25_i-HJwOFEf/view?usp=drive_link) | [link](https://drive.google.com/file/d/1G8JyovxzMqD_M2_Iytd8VhCHntEM87u6/view?usp=drive_link) | [link](https://drive.google.com/file/d/1PRyG0Sbi65v9rO6RGMRqb5jO3IKQ6WJP/view?usp=drive_link) |

### Road Surface Elevation Estimation

| Model | `rgb` | `polar-rgb` | `mono-polar` | `tri-polar` | `rgb-mono` | `rgb-tri` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| RoadBEV | [link](https://drive.google.com/file/d/1ioLCrQ9K-JFlKrFOa21qUAg3DLHaXL-2/view?usp=drive_link) | [link](https://drive.google.com/file/d/1rv-xtrkpC03bKgrSiMKsDf_rErnDxgo4/view?usp=drive_link) | [link](https://drive.google.com/file/d/1PaRUklUGm7FdxBR9JANmGU2LVvEIMlHi/view?usp=drive_link) | [link](https://drive.google.com/file/d/19MA5_BhSV1m63fIy3z45G3Ud2U3Pi8sz/view?usp=drive_link) | [link](https://drive.google.com/file/d/1AJ4PuN2taYftfRK1reap_QI7mJ_3V9UY/view?usp=drive_link) | [link](https://drive.google.com/file/d/14H1OoH_1CQn4uU1Q-7EhBknHuzrU1yhA/view?usp=drive_link) |

## Citation<a name="citation"></a>
```bibtex
@inproceedings{prism2026,
  title     = {PRISM: Polarimetric Road-surface Intelligent Sensing and Measurement Dataset},
  author    = {Jung, Minseo and Kim, Chanwoo and Kim, Donggue and Kim, Hyunjun and Park, Jeongwoo and Jo, Kichun},
  booktitle = {Advances in Neural Information Processing Systems 39 (NeurIPS 2026), Evaluations and Datasets Track},
  year      = {2026}
}
```

## License<a name="license"></a>
This project is licensed under the Apache-2.0 License - see the [Apache 2.0 license](./LICENSE) file for details.

## Acknowledgement<a name="acknowledgement"></a>
This work was supported in part by the Vehicle Control Technology Development Team 3, Hyundai Motor Company and in part by the National Research Foundation of Korea (NRF) funded by the Korean Government Ministry of Science and ICT (MSIT) under grant No. RS-2024-00421129.

This project is not possible without the following codebases.
* [RSRD](https://github.com/ztsrxh/RSRD_dev_toolkit.git)
* [RoadBEV](https://github.com/ztsrxh/RoadBEV.git)