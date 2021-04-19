import argparse
import os
import subprocess

for data in ['cifar100', 'svhn', 'imagenet', 'gaussian', 'square', 'reso']:
    subprocess.run(f"python train.py --gpu 0 --ood {data} --out results/run_{data}_try1", shell=True)
