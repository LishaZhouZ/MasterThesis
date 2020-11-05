import time
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from model_utility import loss_l2, loss_l1, PSNRMetric, MS_SSIMMetric, WaveletConvLayer, WaveletInvLayer
import datetime
from pathlib import Path
import glob
from PIL import Image
import math
#with reg loss
#@tf.function
# def grad(model, images, labels, optimizer):
#     with tf.GradientTape() as tape:
#         output = model(images, training = True)
#         reconstructed = images + output
#         #reconstructed = tf.clip_by_value(images + output, clip_value_min=0., clip_value_max=255.)
#         loss_RGB = loss_fn(reconstructed, labels)
#         reg_losses = tf.math.add_n(model.losses)
#         total_loss = loss_RGB + reg_losses
#     grads = tape.gradient(total_loss, model.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))

#     return loss_RGB, reg_losses, total_loss, reconstructed

# without reg loss
@tf.function
def grad_l2(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss_RGB = loss_l2(output, labels)
    grads = tape.gradient(loss_RGB, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_RGB, output

@tf.function
def grad_l1(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss_RGB = loss_l1(output, labels)
    grads = tape.gradient(loss_RGB, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_RGB, output

def train_one_epoch(model1, model2, dataset, optimizer, logdir, ckpt, manager, record_step):
    org_psnr = PSNRMetric()
    opt_psnr = PSNRMetric()
    
    avg_loss1 = tf.keras.metrics.Mean()
    avg_loss2 = tf.keras.metrics.Mean()
    
    reg_loss1 = tf.keras.metrics.Mean()
    reg_loss2 = tf.keras.metrics.Mean()

    train_writer = tf.summary.create_file_writer(logdir + "/train")
    

    for images, labels in dataset.take(800):
        wave_convert = WaveletConvLayer()
        wave_inverse = WaveletInvLayer()

        #input_wav = wave_convert(images)
        labels_wav = wave_convert(labels)

        loss_LL, reconstructed_LL = grad_l2(model1, images, labels_wav[:,:,:,0:3], optimizer)
        loss_LLHH, reconstructed_LLHH = grad_l1(model2, images, labels_wav[:,:,:,3:12], optimizer)
        
        reconstructed = wave_inverse(tf.concat([reconstructed_LL, reconstructed_LLHH], 3))

        reg_losses1 = tf.math.add_n(model1.losses)
        reg_losses2 = tf.math.add_n(model2.losses)


        org_psnr(images, labels)
        opt_psnr(reconstructed, labels)
        #avg_loss(loss_RGB)
        avg_loss1(loss_LL)
        avg_loss2(loss_LLHH)
        
        reg_loss1(reg_losses1)
        reg_loss2(reg_losses2)

        step = ckpt.step.numpy()
        if int(step) % record_step == 0:
            avg_relative_psnr = opt_psnr.result() - org_psnr.result()
            print("Step " + str(step) + " loss_LL {:1.2f},".format(avg_loss1.result())
                                                                + " loss_LLHH {:1.2f},".format(avg_loss2.result()) 
                                                                + " train_psnr {:1.5f},".format(opt_psnr.result())
                                                                + " org_psnr {:1.5f},".format(org_psnr.result())
                                                                + " gain {:1.5f}".format(avg_relative_psnr))
            #for record
            with train_writer.as_default():
                tf.summary.scalar('optimizer_lr_t', optimizer.learning_rate, step = step)
                tf.summary.scalar('LL_regLoss', reg_loss1.result(), step = step)
                tf.summary.scalar('LLHH_regloss', reg_loss2.result(), step = step)
                tf.summary.scalar('train_psnr', opt_psnr.result(), step = step)
                tf.summary.scalar('relative_tr_psnr', avg_relative_psnr, step = step)
            
            org_psnr.reset_states()
            opt_psnr.reset_states()
            
            avg_loss1.reset_states()
            avg_loss2.reset_states()
            reg_loss1.reset_states()
            reg_loss2.reset_states()

        ckpt.step.assign_add(1)


def evaluate_model(model1, model2, logdir, epoch, dir_input = Path('/mnt/data4/Students/Lisha/images/validation/live1_0-100/qp10'), 
                dir_label = Path('/mnt/data4/Students/Lisha/images/validation/live1_gt')):

    filepaths_label = sorted(dir_label.glob('*'))
    filepaths_input = sorted(dir_input.glob('*'))

    org_psnr = np.zeros(len(filepaths_label))
    rec_psnr = np.zeros(len(filepaths_label))
    
    org_ssim = np.zeros(len(filepaths_label))
    rec_ssim = np.zeros(len(filepaths_label))
    
    test_writer = tf.summary.create_file_writer(logdir + "/test")
    
    for i in range(len(filepaths_label)):
        img_label = Image.open(filepaths_label[i])
        img_input = Image.open(filepaths_input[i])
         
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
        

        wave_inverse = WaveletInvLayer()

        output1 = model1(img_s_input_batch, training= False)
        output2 = model2(img_s_input_batch, training= False)
        output = wave_inverse(tf.concat([output1, output2],3))
        
        output_cut = tf.slice(output, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])


        org_psnr[i] = tf.image.psnr(img_s_label, img_s_input, 255.0).numpy()
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
        
        
 
    