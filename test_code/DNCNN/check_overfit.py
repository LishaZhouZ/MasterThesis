import sys
sys.path.insert(1, '/home/ge29nab/MasterThesis/DnCNN')
import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import timeit
from pathlib import Path
import glob
from PIL import Image
import numpy as np
import math
#import DnCNN_Feature_Attention
import pandas as pd
from tensorflow import config as config
print("Num GPUs Available: ", len(config.experimental.list_physical_devices('GPU')))
gpus = config.experimental.list_physical_devices('GPU')
config.experimental.set_visible_devices(gpus[2], 'GPU')
logical_gpus = config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckptPath', dest='restore_ckptPath', type=str,default='/mnt/data4/Students/Lisha/tf_ckpts/DnCNN_test')
parser.add_argument('--CPU', dest='CPU', type = bool, default = False)
args = parser.parse_args()


import tensorflow as tf
import model_DnCNN


def check():
    
    dir_label = Path('/mnt/data4/Students/Lisha/images/train/groundtruth')
    dir_input = Path('/mnt/data4/Students/Lisha/images/train/qp0-100/qp10')
    
    filepaths_label = sorted(dir_label.glob('*'))
    numDebug = 1000
    filepaths_label = filepaths_label[:numDebug]
    filenames = [item.name[0:-4] + '.jpg' for item in filepaths_label]

    model = model_DnCNN.DnCNN()

    
    train_writer = tf.summary.create_file_writer('/home/ge29nab/MasterThesis/logs/DnCNN_test/train')
    original_writer = tf.summary.create_file_writer('/home/ge29nab/MasterThesis/logs/DnCNN_test/original')
#---------------------------------------------------------------------------------------
    org_psnr = np.zeros(len(filepaths_label))
    rec_psnr = np.zeros(len(filepaths_label))
    
    org_ssim = np.zeros(len(filepaths_label))
    rec_ssim = np.zeros(len(filepaths_label))

    for epoch in range(1,41):
        ckptPath = '/mnt/data4/Students/Lisha/tf_ckpts/DnCNN_test/ckpt-'+str(epoch)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = model)
        ckpt.restore(ckptPath)#tf.train.latest_checkpoint(args.restore_ckptPath))
        print("Successfully restore from %s"%ckptPath)
        for i in range(len(filepaths_label)):

            img_label = Image.open(filepaths_label[i])
            img_input = Image.open(Path(dir_input, filenames[i]))
            a = np.array(img_label, dtype="float32")
            b = np.array(img_input, dtype="float32")
            img_s_label = tf.convert_to_tensor(a)
            img_s_input = tf.convert_to_tensor(b)
            #padding
            shape_input = tf.shape(img_s_input).numpy()
            padding_up = math.ceil(6-shape_input[0]%6/2)
            padding_down = math.floor(6-shape_input[0]%6/2)
            padding_left = math.ceil(6-shape_input[1]%6/2)
            padding_right = math.floor(6-shape_input[1]%6/2)
            paddings = tf.constant([[padding_up, padding_down,], [padding_left, padding_right], [0, 0]])
            img_s_input_padded = tf.pad(img_s_input, paddings, "REFLECT")
            img_s_input_batch = tf.expand_dims(img_s_input_padded, axis = 0)
            img_s_label_batch = tf.expand_dims(img_s_label, axis = 0)
            #start = timeit.default_timer()
            output = model(img_s_input_batch)
            output_cut = tf.slice(output, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])

            org_psnr[i] = tf.image.psnr(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0).numpy()
            rec_psnr[i] = tf.image.psnr(output_cut, img_s_label_batch, 255.0).numpy()
            org_ssim[i] = tf.image.ssim_multiscale(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0)
            rec_ssim[i] = tf.image.ssim_multiscale(output_cut, img_s_label_batch, 255.0)
            print('Image ' + str(i) + ' org_psnr:%.4f,' % org_psnr[i] + 'after_psnr:%.4f,' % rec_psnr[i], ' org_ssim:%.4f,' % org_ssim[i] + 'after_ssim:%.4f' % rec_ssim[i])

        print("Epoch " + str(epoch) + " val_psnr {:1.5f},".format(rec_psnr.mean())
                            + " org_psnr {:1.5f},".format(org_psnr.mean())
                            + " gain {:1.5f}".format(rec_psnr.mean()-org_psnr.mean())
                            + " msssim {:1.5f}".format(rec_ssim.mean()))

        with train_writer.as_default():
            tf.summary.scalar('psnr_gain', rec_psnr.mean()-org_psnr.mean(), step=epoch)
            tf.summary.scalar('psnr', rec_psnr.mean(), step=epoch)
            tf.summary.scalar('msssim', rec_ssim.mean(), step=epoch)
        
        with original_writer.as_default():
            tf.summary.scalar('psnr', org_psnr.mean(), step=epoch)
            tf.summary.scalar('msssim', org_ssim.mean(), step=epoch)

if __name__ == '__main__':
    check()