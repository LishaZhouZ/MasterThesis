# this file restore the model and checkpoint from file, and to check resulting image
import models
import os
from pathlib import Path
import matplotlib.pyplot as plt
import glob
from PIL import Image
import numpy as np
import math
import tensorflow as tf
from utils_py3_tfrecord_128 import save_images
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
#GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


if __name__ == "__main__":

    dir_label = Path('/mnt/data4/Students/Lisha/images/validation/live1_gt')
    dir_input = Path('/mnt/data4/Students/Lisha/images/validation/live1_0-100/qp10')

    filepaths_label = sorted(dir_label.glob('*'))
    filepaths_input = sorted(dir_input.glob('*'))

    
    restore_ckptPath = Path('/mnt/data4/Students/Lisha/tf_ckpts/DualinNet_bz64/ckpt-35')
    mirror = 4
    model = models.DualinNet()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = model)
    ckpt.restore(str(restore_ckptPath))#tf.train.latest_checkpoint(args.restore_ckptPath))
    print("Successfully restore from %s"%restore_ckptPath)
    
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

        output = model(img_s_input_batch, training = False)

        output_cut = tf.slice(output, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])

        org_psnr = tf.image.psnr(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0).numpy()
        rec_psnr = tf.image.psnr(output_cut, img_s_label_batch, 255.0).numpy()
        org_ssim= tf.image.ssim_multiscale(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0)
        rec_ssim = tf.image.ssim_multiscale(output_cut, img_s_label_batch, 255.0)
        print('Image ' +str(i)+ ' org_psnr:%.4f,' % org_psnr + 'after_psnr:%.4f,' % rec_psnr, ' org_ssim:%.4f,' % org_ssim + 'after_ssim:%.4f' % rec_ssim)
        
        save_images(Path('/mnt/data4/Students/Lisha/outcome/DualinNet-fm64-10-bz64/' + str(i) +'.bmp'), img_s_label, noisy_image = img_s_input.numpy(), clean_image = np.squeeze(output_cut, axis=0))
    
    #output = model.predict(img_s_input_batch)
    #reconstructed = img_s_input_batch + output

    #reconstructed_s = np.squeeze(reconstructed, axis=0)

    #print(tf.image.psnr(img_s_label, img_s_input, 255))
    #print(tf.image.psnr(reconstructed_s, img_s_label, 255))

    # show RGB one channel
    # for images, labels in train_dataset.take(1):
    #     images = tf.image.grayscale_to_rgb(images[0,:,:,:])
    #     temp = np.zeros(images.shape, dtype='uint8')
    #     temp[:,:,0] = images[:,:,0]
    #     plt.imshow(temp)
    #     plt.show()


    # f = plt.figure()
    
    # f.add_subplot(1,3,1)
    # plt.imshow(img_s_label/255)
    # plt.axis('off')
    # plt.title('original')

    # f.add_subplot(1,3, 2)
    # plt.imshow(reconstructed_s/255)
    # plt.axis('off')
    # plt.title('reconstructed')

    # f.add_subplot(1,3, 3)
    # plt.imshow(img_s_input/255)
    # plt.axis('off')
    # plt.title('input')

    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=None, wspace = .001)
    # plt.show()
    #save_images(Path('/home/lisha/Forschungspraxis/images/test/outcome/' + str(number) + '.bmp'), img_s_label, noisy_image = img_s_input, clean_image = reconstructed_s)