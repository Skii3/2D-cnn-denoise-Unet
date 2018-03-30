from easydict import EasyDict as edict
import json
import numpy as np
import os

configPara = edict()

configPara.if_continue_train=False

configPara.nn1=128
configPara.nn2=128
configPara.rate=4

configPara.lr_init = 2e-4
configPara.beta1 = 0.5
configPara.n_epoch =10000

configPara.L1_lambda=1
configPara.tvDiff_lambda=200
configPara.loop_lambda=1
configPara.gan_lambda=0.01

configPara.test_freq=20
configPara.save_model_freq=20

configPara.scale= 5
configPara.if_scale=True

nn1=configPara.nn1
nn2=configPara.nn2
scale=configPara.scale
tvDiff_lambda=configPara.tvDiff_lambda
configPara.samples_save_dir = "train_result/samples9/"
configPara.test_save_dir="train_result/test_result9"
configPara.checkpoint_dir = "train_result/checkpoint9/"
configPara.buffer_dir="train_result/buffer9/"

configPara.train_image_path="train_Data/"
#configPara.test_image_clean_path="test_data/clean/"
configPara.test_image_path="test_data_txt/input/"

if not os.path.exists(configPara.checkpoint_dir):
    os.makedirs(configPara.checkpoint_dir)
if not os.path.exists(configPara.test_save_dir):
    os.makedirs(configPara.test_save_dir)
if not os.path.exists(configPara.samples_save_dir):
    os.makedirs(configPara.samples_save_dir)
if not os.path.exists(configPara.buffer_dir):
    os.makedirs(configPara.buffer_dir)
