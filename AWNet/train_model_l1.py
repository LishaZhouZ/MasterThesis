import time
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from model_utility import loss_l1, PSNRMetric, MS_SSIMMetric
import datetime
from pathlib import Path
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
def grad(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        #reconstructed = tf.clip_by_value(images + output, clip_value_min=0., clip_value_max=255.)
        loss_RGB = loss_l1(output, labels)
    grads = tape.gradient(loss_RGB, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_RGB, output

def train_one_epoch(model, dataset, optimizer, writer, ckpt, manager, record_step):
    org_psnr = PSNRMetric()
    opt_psnr = PSNRMetric()
    avg_loss = tf.keras.metrics.Mean()
    rgb_loss = tf.keras.metrics.Mean()
    reg_loss = tf.keras.metrics.Mean()
    for images, labels in dataset.take(10):
        loss_RGB, reconstructed = grad(model, images, labels, optimizer)
        #loss_RGB, reg_losses, total_loss, reconstructed = grad(model, images, labels, optimizer)
        reg_losses = tf.math.add_n(model.losses)
        total_loss = loss_RGB + reg_losses

        org_psnr(images, labels)
        opt_psnr(reconstructed, labels)
        #avg_loss(loss_RGB)
        avg_loss(total_loss)
        rgb_loss(loss_RGB)
        reg_loss(reg_losses)

        step = ckpt.step.numpy()

        if int(step) % record_step == 0:
            avg_relative_psnr = opt_psnr.result() - org_psnr.result()
            print("Step " + str(step) + " loss {:1.2f},".format(avg_loss.result()) 
                                                                + " train_psnr {:1.5f},".format(opt_psnr.result())
                                                                + " org_psnr {:1.5f},".format(org_psnr.result())
                                                                + " gain {:1.5f}".format(avg_relative_psnr))
            #for record
            with writer.as_default():
                tf.summary.scalar('optimizer_lr_t', optimizer.learning_rate, step = step)
                tf.summary.scalar('train_loss', avg_loss.result(), step = step)
                tf.summary.scalar('lossRGB', rgb_loss.result(), step = step)
                tf.summary.scalar('reg_loss', reg_loss.result(), step = step)
                tf.summary.scalar('train_psnr', opt_psnr.result(), step = step)
                tf.summary.scalar('original_psnr', org_psnr.result(), step = step)
                tf.summary.scalar('relative_tr_psnr', avg_relative_psnr, step = step)
            
            org_psnr.reset_states()
            opt_psnr.reset_states()
            avg_loss.reset_states()
            rgb_loss.reset_states()
            reg_loss.reset_states()

        ckpt.step.assign_add(1)


def evaluate_model(model, writer, epoch, dir_input = Path('/mnt/data4/Students/Lisha/images/validation/live1_0-100/qp10'), 
                dir_label = Path('/mnt/data4/Students/Lisha/images/validation/live1_gt')):

    filepaths_label = sorted(dir_label.glob('*'))
    filepaths_input = sorted(dir_input.glob('*'))

    org_psnr = np.zeros(len(filepaths_label))
    rec_psnr = np.zeros(len(filepaths_label))
    
    org_ssim = np.zeros(len(filepaths_label))
    rec_ssim = np.zeros(len(filepaths_label))

    for i in range(len(filepaths_label)):
        img_label = Image.open(filepaths_label[i])
        img_input = Image.open(filepaths_input[i])
         
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
        
        output = model.predict(img_s_input_batch)
        
        output_cut = tf.slice(output, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])


        org_psnr[i] = tf.image.psnr(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0).numpy()
        rec_psnr[i] = tf.image.psnr(output_cut, img_s_label_batch, 255.0).numpy()
        org_ssim[i] = tf.image.ssim_multiscale(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0)
        rec_ssim[i] = tf.image.ssim_multiscale(output_cut, img_s_label_batch, 255.0)
    
    print("Epoch " + str(epoch) + " val_psnr {:1.5f},".format(rec_psnr.mean())
                            + " org_psnr {:1.5f},".format(org_psnr.mean())
                            + " gain {:1.5f}".format(rec_psnr.mean()-org_psnr.mean())
                            + " msssim {:1.5f}".format(rec_ssim.mean()))
    with writer.as_default():
        tf.summary.scalar('relative_val_psnr', rec_psnr.mean()-org_psnr.mean(), step=epoch)
        tf.summary.scalar('validation_psnr', rec_psnr.mean(), step=epoch)
        tf.summary.scalar('validation_msssim', rec_ssim.mean(), step=epoch)
        
 
    


