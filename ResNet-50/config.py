import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime

now = datetime.now()

parser = argparse.ArgumentParser()

# Input
parser.add_argument('--img_w', default=256, help='image width')
parser.add_argument('--img_h', default=256, help='image height')
parser.add_argument('--n_ch', default=1, help='number of input channels')
parser.add_argument('--n_cls', default=14, help='number of conditions')

# Training and network configs
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument('--batch_size', default=16, help='Training batch size')
parser.add_argument('--num_epoch', default=200, help='Total number of training epochs')
parser.add_argument('--stddev', default=0.01, help='std for W initializer')
parser.add_argument('--lmbda', default=5e-04, help='Regularization coefficient')
parser.add_argument('--init_lr', default=0.001, help='Initial learning rate')

# Validation
parser.add_argument('--val_batch_size', default=100, help='Validation batch size')

# Save and display
parser.add_argument('--report_freq', default=100, help='The frequency of displaying train results (step)')
save_dir = './checkpoints/' + now.strftime("%Y%m%d-%H%M%S")
parser.add_argument('--logs_path', default="./graph/" + now.strftime("%Y%m%d-%H%M%S"), help='path to save logs')
parser.add_argument('--save_dir', default='./checkpoints/' + now.strftime("%Y%m%d-%H%M%S"), help='path to save models')
parser.add_argument('--results_dir', default='./results/' + now.strftime("%Y%m%d-%H%M%S"), help='path to save results')
# parser.add_argument('--load_dir', default='./checkpoints/', help='path to load models')

args = parser.parse_args()

