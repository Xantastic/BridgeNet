# BridgeNet: A Unified Multimodal Framework for Bridging 2D and 3D Industrial Anomaly Detection
An Xiang*, Zixuan Huang*, Xitong Gao*, Kejiang Ye†, Cheng-zhong Xu (* Equal contribution; † Corresponding authors)

Our paper has been accepted by ACM MM 2025 [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3746027.3755261).
![](figures/bridgenet.png)

## Quick Start

### 1. Environment Installation

#### Prerequisites
- Python 3.9
- Conda package manager

#### Option A: Automatic Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/Bridgenet-3D/BridgeNet.git
cd BridgeNet

# Create conda environment (environment name is 'bridgenet')
conda env create -f environment.yml

# Activate environment
conda activate bridgenet
```

#### Option B: Manual Installation

```bash
# Create new conda environment
conda create -n bridgenet python=3.9
conda activate bridgenet

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
pip install ...
```

### 2. Dataset Structure

The dataset should be organized in the following structure:

```
mvtec_process/
├── cookie/
│   ├── train/
│   │   └── good/
│   │       ├── 000.png
│   │       ├── 001.png
│   │       └── ...
│   ├── test/
│   │   ├── good/
│   │   │   ├── 000.png
│   │   │   └── ...
│   │   ├── crack/
│   │   │   ├── 000.png
│   │   │   └── ...
│   │   ├── contamination/
│   │   │   ├── 000.png
│   │   │   └── ...
│   │   └── ...
│   └── ground_truth/
│       ├── crack/
│       │   ├── 000.png
│       │   └── ...
│       ├── contamination/
│       │   ├── 000.png
│       │   └── ...
│       └── ...
├── dowel/
│   ├── train/
│   │   └── good/
│   │       ├── 000.png
│   │       ├── 001.png
│   │       └── ...
│   ├── test/
│   │   ├── good/
│   │   │   ├── 000.png
│   │   │   └── ...
│   │   ├── bent/
│   │   │   ├── 000.png
│   │   │   └── ...
│   │   ├── cut/
│   │   │   ├── 000.png
│   │   │   └── ...
│   │   └── ...
│   └── ground_truth/
│       ├── bent/
│       │   ├── 000.png
│       │   └── ...
│       ├── cut/
│       │   ├── 000.png
│       │   └── ...
│       └── ...
├── ...
│      
└── Depth
```

Each category (e.g., cookie, dowel) contains:
- `train/good/`: Normal training samples
- `test/good/`: Normal test samples
- `test/<anomaly>/`: Anomalous test samples
- `ground_truth/<anomaly>/`: Ground truth masks for anomalies

The dataset formatted [[mvtec3d_formatted]](https://pan.baidu.com/s/1K-mHeK-clHmBru82U5R90g?pwd=6p7f)

Original dataset [[mvtec3d]](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)

### 3. Running BridgeNet

We provide a simple shell script to run BridgeNet on the MVTec-3D dataset:

```bash
cd shell
bash run-mvtec.sh
```

### 4. Results

Results will be saved in the `results/` directory with the following structure:
/root/3D/BridgeNet-main_/results/models/backbone_0/mvtec3d_foam
```
results/
├── models/
│   └── backbone_0/
│       └── class_name/
├── eval
│   └── class_name/
└── training/            
    └── class_name/            
            
```

## Citation

If you use BridgeNet in your research, please cite:

```bibtex
@inproceedings{xiang2025bridgenet,
  title={BridgeNet: A Unified Multimodal Framework for Bridging 2D and 3D Industrial Anomaly Detection},
  author={Xiang, An and Huang, Zixuan and Gao, Xitong and Ye, Kejiang and Xu, Cheng-zhong},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={1579--1587},
  year={2025}
}
```
## Acknowledgments

We would like to thank the following repositories for their valuable contributions and support:

- [GLASS](https://github.com/cqylunlun/GLASS)
- [3DSR](https://github.com/VitjanZ/3DSR)

These open-source projects have been instrumental in advancing the field of anomaly detection.
