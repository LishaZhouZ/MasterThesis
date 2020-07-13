import sys
sys.path.insert(1, '../')
import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import timeit
from pathlib import Path
import matplotlib.pyplot as plt
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
parser.add_argument('--ckptPath', dest='restore_ckptPath', type=str,default='/home/ge29nab/MasterThesis/tf_ckpts/DNCNN_retrain2/ckpt-89')
parser.add_argument('--model', dest='model', type = str,default="DnCNN", help='RIDNet,DnCNN')
parser.add_argument('--CPU', dest='CPU', type = bool, default = False)
args = parser.parse_args()


import tensorflow as tf
import model_DnCNN


if __name__ == "__main__":

    qorg_psnr = np.zeros(21)
    qrec_psnr = np.zeros(21)
    
    qorg_ssim = np.zeros(21)
    qrec_ssim = np.zeros(21)
    qtimesum = np.zeros(21)
    qtimemean = np.zeros(21)    
    
    dir_label = Path('/mnt/data4/Students/Lisha/images/validation/live1_gt')
    dir_input = Path('/mnt/data4/Students/Lisha/images/validation/live1_0-100')
    
    filepaths_label = sorted(dir_label.glob('*'))
    filenames = [item.name[0:-4] + '.jpg' for item in filepaths_label]
    q_input = []
    for qulaity in range(0,101,5):
        q_input.append(Path(dir_input, Path('qp'+str(qulaity))))
    
    if args.model == "DnCNN":
        model = model_DnCNN.DnCNN()
    elif args.model == "RIDNet":
        model = DnCNN_Feature_Attention.RIDNet()
    else:
        print('Error model name!')

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = model)
    ckpt.restore(args.restore_ckptPath)#tf.train.latest_checkpoint(args.restore_ckptPath))
    print("Successfully restore from %s"%args.restore_ckptPath)

#---------------------------------------------------------------------------------------
    org_psnr = np.zeros(len(filepaths_label))
    rec_psnr = np.zeros(len(filepaths_label))
    
    org_ssim = np.zeros(len(filepaths_label))
    rec_ssim = np.zeros(len(filepaths_label))
    time = np.zeros(len(filepaths_label))
    
    for q in range(13,14):
        for i in range(len(filepaths_label)):

            img_label = Image.open(filepaths_label[i])
            img_input = Image.open(Path(q_input[q], filenames[i]))

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


            #stop = timeit.default_timer()
            #time[i] = stop - start

            output_cut = tf.slice(output, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])


            org_psnr[i] = tf.image.psnr(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0).numpy()
            rec_psnr[i] = tf.image.psnr(output_cut, img_s_label_batch, 255.0).numpy()
            org_ssim[i] = tf.image.ssim_multiscale(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0)
            rec_ssim[i] = tf.image.ssim_multiscale(output_cut, img_s_label_batch, 255.0)
            print('Image ' + str(i) + ' org_psnr:%.4f,' % org_psnr[i] + 'after_psnr:%.4f,' % rec_psnr[i], ' org_ssim:%.4f,' % org_ssim[i] + 'after_ssim:%.4f' % rec_ssim[i])


        
    for q in range(len(q_input)):
        for i in range(len(filepaths_label)):

            img_label = Image.open(filepaths_label[i])
            img_input = Image.open(Path(q_input[q], filenames[i]))

            a = np.array(img_label, dtype="float32")
            b = np.array(img_input, dtype="float32")
            img_s_label = tf.convert_to_tensor(a)
            img_s_input = tf.convert_to_tensor(b)

            #padding
            shape_input = tf.shape(img_s_input).numpy()
            padding_up = math.ceil(16-shape_input[0]%16/2)
            padding_down = math.floor(16-shape_input[0]%16/2)
            padding_left = math.ceil(16-shape_input[1]%16/2)
            padding_right = math.floor(16-shape_input[1]%16/2)
            paddings = tf.constant([[padding_up, padding_down,], [padding_left, padding_right], [0, 0]])

            img_s_input_padded = tf.pad(img_s_input, paddings, "REFLECT")
            img_s_input_batch = tf.expand_dims(img_s_input_padded, axis = 0)

            img_s_label_batch = tf.expand_dims(img_s_label, axis = 0)

            start = timeit.default_timer()

            output = model(img_s_input_batch)

            stop = timeit.default_timer()
            time[i] = stop - start

            output_cut = tf.slice(output, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])

            org_psnr[i] = tf.image.psnr(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0).numpy()
            rec_psnr[i] = tf.image.psnr(output_cut, img_s_label_batch, 255.0).numpy()
            org_ssim[i] = tf.image.ssim_multiscale(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0)
            rec_ssim[i] = tf.image.ssim_multiscale(output_cut, img_s_label_batch, 255.0)
            print('Image ' + str(i) + ' org_psnr:%.4f,' % org_psnr[i] + 'after_psnr:%.4f,' % rec_psnr[i], ' org_ssim:%.4f,' % org_ssim[i] + 'after_ssim:%.4f' % rec_ssim[i])


            #save_images(Path('/home/lisha/Forschungspraxis/outcome/m2/' + str(i) + '-5.bmp'), img_s_label, noisy_image = img_s_input.numpy(), clean_image = np.squeeze(output_cut, axis=0))
        
    
        qorg_psnr[q] = np.mean(org_psnr)
        qrec_psnr[q] = np.mean(rec_psnr)
        qorg_ssim[q] = np.mean(org_ssim)
        qrec_ssim[q] = np.mean(rec_ssim)
        qtimesum[q] = np.sum(time)
        qtimemean[q] = np.mean(time)
        print('For quality %d:' % q)
        print('average org_psnr:%.4f' % qorg_psnr[q])
        print('average after_psnr:%.4f' % qrec_psnr[q])
        print('average org_ssim:%.4f' % qorg_ssim[q] )
        print('average after_ssim:%.4f' % qrec_ssim[q])
        print('time %.4f' %qtimesum[q])
        print('time %.4f' %qtimemean[q])
  
    dataCollection = np.vstack((qorg_psnr, qrec_psnr, qorg_ssim, qrec_ssim, qtimesum, qtimemean)).T
    df = pd.DataFrame(dataCollection, columns = ['org_psnr','rec_psnr','org_ssim','rec_ssim','qtimesum','qtimemean'])
    df.to_csv("ResultDNCNN_GPU.xlsx")
