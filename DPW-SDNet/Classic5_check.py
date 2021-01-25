# this file restore the model and checkpoint from file, and to check resulting image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import glob
from PIL import Image
import numpy as np
import math
import tensorflow as tf
import models
import timeit

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
#GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#CPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":
    #live1
    #dir_label = Path('/mnt/data4/Students/Lisha/images/validation/live1_gt')
    #dir_input = Path('/mnt/data4/Students/Lisha/images/validation/live1_0-100/qp10')
    
    #classic 5
    dir_label = Path('/mnt/data4/Students/Lisha/images/validation/groundtruth_5')
    dir_input = Path('/mnt/data4/Students/Lisha/images/validation/compressed_Q10_5')
    restore_ckptPath = Path('/mnt/data4/Students/Lisha/tf_ckpts/DPW-SDNet/ckpt-31')
    
    filepaths_label = sorted(dir_label.glob('*'))
    filepaths_input = sorted(dir_input.glob('*'))
    time = np.zeros(len(filepaths_label))

    org_psnr = np.zeros(len(filepaths_label))
    org_ssim = np.zeros(len(filepaths_label))
    rec_psnr = np.zeros(len(filepaths_label))
    rec_ssim = np.zeros(len(filepaths_label))

    #dir_label = Path('./images/test/groundtruth_5')
    #dir_input = Path('./images/test/compressed_Q10_5')

    model = models.DPW_SDNet()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = model)
    ckpt.restore(str(restore_ckptPath))#tf.train.latest_checkpoint(args.restore_ckptPath))
    print("Successfully restore from %s"%restore_ckptPath)

    mirror = 4
    for i in range(5):
        img_label = Image.open(filepaths_label[i])
        img_input = Image.open(filepaths_input[i])
        
        a = np.array(img_label, dtype="float32")
        b = np.array(img_input, dtype="float32")
            
        img_s_label = tf.convert_to_tensor(a)
        img_s_input = tf.convert_to_tensor(b)#padding
            
        shape_input = tf.shape(img_s_input).numpy()
        padding_up = math.ceil(mirror-shape_input[0]%mirror/2)
        padding_down = math.floor(mirror-shape_input[0]%mirror/2)
        padding_left = math.ceil(mirror-shape_input[1]%mirror/2)
        padding_right = math.floor(mirror-shape_input[1]%mirror/2)
        paddings = tf.constant([[padding_up, padding_down,], [padding_left, padding_right], [0, 0]])
        img_s_input_padded = tf.pad(img_s_input, paddings, "REFLECT")
        img_s_input_batch = tf.expand_dims(img_s_input_padded, axis = 0)
        img_s_label_batch = tf.expand_dims(img_s_label, axis = 0)
        
        #start = timeit.default_timer()
        output = model(img_s_input_batch, training = False)
        #stop = timeit.default_timer()
        #time[i] = stop - start

        output_cut = tf.slice(output, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])

        org_psnr[i] = tf.image.psnr(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0).numpy()
        rec_psnr[i] = tf.image.psnr(output_cut, img_s_label_batch, 255.0).numpy()
        org_ssim[i] = tf.image.ssim_multiscale(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0)
        rec_ssim[i] = tf.image.ssim_multiscale(output_cut, img_s_label_batch, 255.0)
        print('Image ' + str(i) + ' org_psnr:%.4f,' % org_psnr[i] + 'after_psnr:%.4f,' % rec_psnr[i], ' org_ssim:%.4f,' % org_ssim[i] + 'after_ssim:%.4f' % rec_ssim[i])

    for i in range(len(filepaths_label)):
        img_label = Image.open(filepaths_label[i])
        img_input = Image.open(filepaths_input[i])
        
        a = np.array(img_label, dtype="float32")
        b = np.array(img_input, dtype="float32")
            
        img_s_label = tf.convert_to_tensor(a)
        img_s_input = tf.convert_to_tensor(b)#padding
            
        shape_input = tf.shape(img_s_input).numpy()
        padding_up = math.ceil(mirror-shape_input[0]%mirror/2)
        padding_down = math.floor(mirror-shape_input[0]%mirror/2)
        padding_left = math.ceil(mirror-shape_input[1]%mirror/2)
        padding_right = math.floor(mirror-shape_input[1]%mirror/2)
        paddings = tf.constant([[padding_up, padding_down,], [padding_left, padding_right], [0, 0]])
        img_s_input_padded = tf.pad(img_s_input, paddings, "REFLECT")
        img_s_input_batch = tf.expand_dims(img_s_input_padded, axis = 0)
        img_s_label_batch = tf.expand_dims(img_s_label, axis = 0)
        
        start = timeit.default_timer()
        output = model(img_s_input_batch, training = False)
        stop = timeit.default_timer()
        time[i] = stop - start

        output_cut = tf.slice(output, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])

        org_psnr[i] = tf.image.psnr(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0).numpy()
        rec_psnr[i] = tf.image.psnr(output_cut, img_s_label_batch, 255.0).numpy()
        org_ssim[i] = tf.image.ssim_multiscale(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0)
        rec_ssim[i] = tf.image.ssim_multiscale(output_cut, img_s_label_batch, 255.0)
        print('Image ' + str(i) + ' org_psnr:%.4f,' % org_psnr[i] + 'after_psnr:%.4f,' % rec_psnr[i], ' org_ssim:%.4f,' % org_ssim[i] + 'after_ssim:%.4f' % rec_ssim[i])

    print("org_psnr %.4f" % np.mean(org_psnr))
    print("org_ssim %.4f"% np.mean(org_ssim))
    print("psnr %.4f" % np.mean(rec_psnr))
    print("ssim %.4f"% np.mean(rec_ssim))
    print("timeSum %.4f" % np.sum(time))
    print("timeMean %.4f"% np.mean(time))