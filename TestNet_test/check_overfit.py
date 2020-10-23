import sys
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import timeit
from pathlib import Path
import glob
from PIL import Image
import numpy as np
import math
import pandas as pd
import tensorflow as tf
import model_utility
import models
#import model_DnCNN

def check(dir_input = Path('/mnt/data4/Students/Lisha/images/validation/live1_0-100/qp10'), 
        dir_label = Path('/mnt/data4/Students/Lisha/images/validation/live1_gt')):
        #logdir = '/home/ge29nab/MasterThesis/logs/', 
        #ckptdir= '/mnt/data4/Students/Lisha/tf_ckpts/',
        #name='TestNet'):
    
    #variants
    model_LL = models.TestNet_LL()
    model_LH = models.TestNet_LH()
    model_HL = models.TestNet_HL()
    model_HH = models.TestNet_HH()

    filepaths_label = sorted(dir_label.glob('*'))
    filenames = [item.name[0:-4] + '.jpg' for item in filepaths_label]

    test_writer = tf.summary.create_file_writer('/home/ge29nab/MasterThesis/logs/TestNet_128_seperate/whole/test')
    original_writer = tf.summary.create_file_writer('/home/ge29nab/MasterThesis/logs/TestNet_128_seperate/whole/original')
#---------------------------------------------------------------------------------------
    org_psnr = np.zeros(len(filepaths_label))
    rec_psnr = np.zeros(len(filepaths_label))
    
    org_ssim = np.zeros(len(filepaths_label))
    rec_ssim = np.zeros(len(filepaths_label))
    for epoch in range(0,40):
        ckpt_LL = tf.train.Checkpoint(step=tf.Variable(1), net = model_LL)
        ckpt_LH = tf.train.Checkpoint(step=tf.Variable(1), net = model_LH)
        ckpt_HL = tf.train.Checkpoint(step=tf.Variable(1), net = model_HL)
        ckpt_HH = tf.train.Checkpoint(step=tf.Variable(1), net = model_HH)

        ckpt_LL.restore('/mnt/data4/Students/Lisha/tf_ckpts/TestNet_128_LL'+ '/ckpt-'+ str(epoch + 1)).expect_partial()
        ckpt_LH.restore('/mnt/data4/Students/Lisha/tf_ckpts/TestNet_128_HL'+ '/ckpt-'+ str(epoch + 1)).expect_partial()
        ckpt_HL.restore('/mnt/data4/Students/Lisha/tf_ckpts/TestNet_128_HL2'+ '/ckpt-'+ str(epoch + 1)).expect_partial()
        ckpt_HH.restore('/mnt/data4/Students/Lisha/tf_ckpts/TestNet_128_HH'+ '/ckpt-'+ str(epoch + 1)).expect_partial()

        print("Successfully restored!")

        for i in range(len(filepaths_label)):
            #read images
            img_label = Image.open(filepaths_label[i])
            img_input = Image.open(Path(dir_input, filenames[i]))

            a = np.array(img_label, dtype="float32")
            b = np.array(img_input, dtype="float32")
            img_s_label = tf.convert_to_tensor(a[:,:,0:3])
            img_s_input = tf.convert_to_tensor(b)

            #padding
            shape_input = tf.shape(img_s_input).numpy()
            padding_up = math.ceil(2-shape_input[0]%2/2)
            padding_down = math.floor(2-shape_input[0]%2/2)
            padding_left = math.ceil(2-shape_input[1]%2/2)
            padding_right = math.floor(2-shape_input[1]%2/2)
            paddings = tf.constant([[padding_up, padding_down,], [padding_left, padding_right], [0, 0]])

            img_s_input_padded = tf.pad(img_s_input, paddings, "REFLECT")
            img_s_input_batch = tf.expand_dims(img_s_input_padded, axis = 0)
            img_s_label_batch = tf.expand_dims(img_s_label, axis = 0)
            #start = timeit.default_timer()
            LL = model_LL(img_s_input_batch, training=False)
            LH = model_LH(img_s_input_batch, training=False)
            HL = model_HL(img_s_input_batch, training=False)
            HH = model_HH(img_s_input_batch, training=False)

            aa = (LL - LH - HL + HH)/2
            bb = (LL - LH + HL - HH)/2
            cc = (LL + LH - HL - HH)/2
            dd = (LL + LH + HL + HH)/2

            concated = tf.concat([aa, bb, cc, dd], 3)
            reconstructed = tf.nn.depth_to_space(concated, 2)

            output_cut = tf.slice(reconstructed, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])

            org_psnr[i] = tf.image.psnr(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0).numpy()
            rec_psnr[i] = tf.image.psnr(output_cut, img_s_label_batch, 255.0).numpy()
            org_ssim[i] = tf.image.ssim_multiscale(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0)
            rec_ssim[i] = tf.image.ssim_multiscale(output_cut, img_s_label_batch, 255.0)
            print("Epoch " + str(epoch) + " val_psnr {:1.5f},".format(rec_psnr.mean())
                            + " org_psnr {:1.5f},".format(org_psnr.mean())
                            + " gain {:1.5f}".format(rec_psnr.mean()-org_psnr.mean())
                            + " msssim {:1.5f}".format(rec_ssim.mean()))

        with test_writer.as_default():
            tf.summary.scalar('psnr_gain', rec_psnr.mean()-org_psnr.mean(), step=epoch)
            tf.summary.scalar('psnr', rec_psnr.mean(), step=epoch)
            tf.summary.scalar('msssim', rec_ssim.mean(), step=epoch)
        
        with original_writer.as_default():
            tf.summary.scalar('psnr', org_psnr.mean(), step=epoch)
            tf.summary.scalar('msssim', org_ssim.mean(), step=epoch)

if __name__ == '__main__':
    check()
