
# Tested configs
# --dataset=mnist --gpus=2 --denoise_train --plot_reconstruction
# --dataset=mnist --gpus=1 --denoise_train --plot_reconstruction --use_gui
# --dataset=svhn --denoise_train --plot_reconstruction --gpus=2 --db_path=dataset/svhn
# --dataset=celebA --denoise_train --plot_reconstruction --gpus=0 --db_path=/ssd_data/CelebA
# --dataset=mnist --gpus=2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no_train', type=bool, default=False)
parser.add_argument('--gpus', type=str, default='')
parser.add_argument('--dataset', type=str, default='celebA')
parser.add_argument('--netname', type=str, default='')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--db_path', type=str, default='')
parser.add_argument('--reg', type=str, default='kl')
parser.add_argument('--denoise_train', dest='denoise_train', action='store_true',
                    help='Use denoise training by adding Gaussian/salt and pepper noise')
parser.add_argument('--plot_reconstruction', dest='plot_reconstruction', action='store_true',
                    help='Plot reconstruction')
parser.add_argument('--use_gui', dest='use_gui', action='store_true',
                    help='Display the results with a GUI window')
parser.add_argument('--vis_frequency', type=int, default=1000,
                    help='How many train batches before we perform visualization')
parser.add_argument('--file_path', type=str, default='models/', help='Where we want to save images and loss values')
parser.add_argument('--num_layers', type=int, default=4, help='Total number of layers in the encoder')

args = parser.parse_args()
fpath = args.file_path
if fpath[-1] != '/':
    fpath += '/'

import matplotlib
if not args.use_gui:
    matplotlib.use('Agg')
else:
    from matplotlib import pyplot as plt
    plt.ion()
    plt.show()

import os
from dataset import *
from vladder import VLadder
from trainer import NoisyTrainer
import numpy as np

if args.gpus is not '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

if args.dataset == 'mnist':
    dataset = MnistDataset()
elif args.dataset == 'lsun':
    dataset = LSUNDataset(db_path=args.db_path)
elif args.dataset == 'celebA':
    dataset = CelebADataset(db_path=args.db_path)
elif args.dataset == 'svhn':
    dataset = SVHNDataset(db_path=args.db_path)
elif args.dataset == 'atari':
    dataset = AtariDataset(db_path=args.db_path)
else:
    print("Unknown dataset")
    exit(-1)

model = VLadder(dataset, file_path=fpath, name=args.netname, reg=args.reg, batch_size=args.batch_size, restart=not args.no_train)
trainer = NoisyTrainer(model, dataset, args)
if args.no_train:
    trainer.output_codes()
else:
    trainer.train()
