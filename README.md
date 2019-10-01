# DeepSBD for ClipShots
This repository contains our implementation of [deepSBD](https://arxiv.org/abs/1705.08214) for [ClipShots](https://github.com/Tangshitao/ClipShots). The code is modified from [here](https://github.com/kenshohara/3D-ResNets-PyTorch).

## Introduction
We implement deepSBD in this repository. There are 2 backbones that can be selected, including the original Alexnet-like and ResNet-18 introduced in our [paper](https://arxiv.org/pdf/1808.04234.pdf).

## Resources
1. The trained model for Alexnet-like backbone. [BaiduYun](https://pan.baidu.com/s/16q3CNuUhLAGkm21PPOqUSg), [Google Drive](https://drive.google.com/open?id=145NCxLhgdrKPIYm-qgp1SRYU_GFmzxxX)

# Traning and Testing
1. Training : opts.py → phase='train'
2. Testing : opts.py → phase='test'
3. Training and Testing : opts.py → phase='full'
4. Run `run_[OS_type].sh`
