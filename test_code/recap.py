# this file restore the model and checkpoint from file, and to check resulting image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import glob
from PIL import Image
import numpy as np
import math
import tensorflow as tf

if __name__ == "__main__":

    dir_label = Path('/home/lisha/Forschungspraxis/images/test/groundtruth_5')
    dir_input = Path('/home/lisha/Downloads/results_bachelor/FastARCNN-5')
    

    filepaths_label = sorted(dir_label.glob('*'))
    filepaths_input = sorted(dir_input.glob('*'))
    time = np.zeros(len(filepaths_label))

    rec_psnr = np.zeros(len(filepaths_label))
    rec_ssim = np.zeros(len(filepaths_label))

    #dir_label = Path('./images/test/groundtruth_5')
    #dir_input = Path('./images/test/compressed_Q10_5')



    for i in range(len(filepaths_label)):
        img_label = Image.open(filepaths_label[i])
        img_input = Image.open(filepaths_input[i])
         
        a = np.array(img_label, dtype="float32")
        b = np.array(img_input, dtype="float32")

        rec_psnr[i] = tf.image.psnr(a,b,max_val=255)
        rec_ssim[i] = tf.image.ssim_multiscale(a,b, max_val=255)
        print(i)

    print("psnr %.4f" % np.mean(rec_psnr))
    print("ssim %.4f"% np.mean(rec_ssim))