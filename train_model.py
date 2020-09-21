import time
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from model_utility import loss_l2, PSNRMetric, MS_SSIMMetric
import datetime

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
        loss_RGB = loss_l2(output, labels)
    grads = tape.gradient(loss_RGB, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_RGB, output

def train_one_epoch(model, dataset, optimizer, writer, ckpt, manager, record_step):
    org_psnr = PSNRMetric()
    opt_psnr = PSNRMetric()
    avg_loss = tf.keras.metrics.Mean()
    rgb_loss = tf.keras.metrics.Mean()
    reg_loss = tf.keras.metrics.Mean()
    for images, labels in dataset:
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


def evaluate_model(model, val_dataset, writer, epoch):
    psnr = PSNRMetric()
    epoch_loss = tf.keras.metrics.Mean()
    ms_ssim = MS_SSIMMetric()
    org_psnr = PSNRMetric()
    
    for images_val, label_val in val_dataset:
        output = model(images_val, training = False)

        # Update val metrics
        loss_RGB = loss_l2(output, label_val)
        reg_losses = tf.math.add_n(model.losses)
        total_loss = loss_RGB + reg_losses
        #record the things
        org_psnr.update_state(label_val, images_val)
        psnr.update_state(label_val, output)
        #epoch_loss.update_state(loss_RGB)
        epoch_loss.update_state(total_loss)
        ms_ssim.update_state(label_val, output)
    
    val_psnr = psnr.result()
    val_loss = epoch_loss.result()
    ms_ssim = ms_ssim.result()
    org_psnr = org_psnr.result()
    gain = val_psnr - org_psnr

    print("Epoch " + str(epoch) + " val_loss {:1.2f},".format(val_loss) 
                            + " val_psnr {:1.5f},".format(val_psnr)
                            + " org_psnr {:1.5f},".format(org_psnr)
                            + " gain {:1.5f}".format(gain)
                            + " msssim {:1.5f}".format(ms_ssim))
    with writer.as_default():
        tf.summary.scalar('relative_val_psnr', gain, step=epoch)
        tf.summary.scalar('validation_loss', val_loss, step=epoch)
        tf.summary.scalar('validation_psnr', val_psnr, step=epoch)
        tf.summary.scalar('validation_msssim', ms_ssim, step=epoch)
        
 
    


