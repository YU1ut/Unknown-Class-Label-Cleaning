# Unknown Class Label Cleaning for Learning with Open-Set Noisy Labels
This is an official PyTorch implementation of Unknown Class Label Cleaning for Learning with Open-Set Noisy Labels. [[Paper]](https://ieeexplore.ieee.org/document/9190652)


## Requirements
- Python 3.7
- PyTorch 1.6.0
- torchvision 0.7.0
- progress
- matplotlib
- numpy

## Preparation
Download TinyImageNet datasets as open-set noise.

```
mkdir data
cd data
wget https://www.dropbox.com/s/1zt54aawvk0245w/Imagenet_resize_full.npy
cd ..
```
Other datasets will be download automatically.

## Usage
Train the network with CIFAR-100 as open-set noise (noise rate = 0.4):
 
```
python train.py --gpu 0 --ood cifar100 --out results/run_cifar100 --percent 0.4
```

The trained model and output will be saved at `results/run_cifar100`.

**For more details and parameters, please refer to --help option.**

Run the experiment with different kind of open-set noise (noise rate = 0.4):

```
python run_all.py
```

## References
- Qing Yu and Kiyoharu Aizawa. "Unknown Class Label Cleaning for Learning with Open-Set Noisy Labels", in ICIP, 2020.