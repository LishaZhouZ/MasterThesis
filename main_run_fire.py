import argparse
from glob import glob
import datetime
import os
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import math
from utils_py3_tfrecord_2 import read_and_decode
from train_model import train_one_epoch, evaluate_model
import model_DnCNN
import DnCNN_Feature_Attention
import fire
import numpy as np

#weigth decay momentum optimizer
#L2 regularization
#tensorboard

def train_process(train_dataset_path = '/mnt/data4/Students/Lisha/patches/train_data.tfrecords', 
    val_dataset_path = '/mnt/data4/Students/Lisha/patches/train_data.tfrecords', 
    lr = 0.0001, ckpt_dir = './tf_ckpts/', name='example', model_type = 'DnCNN', batch_size = 32, epochs = 40):
    train_dataset = read_and_decode(train_dataset_path, batch_size)
    val_dataset = read_and_decode(val_dataset_path, batch_size)
    record_step = 1
    ckpt_directory = ckpt_dir + name
    decay_lr = np.ones(epochs+1)
    decay_lr[0:25]= lr
    #decay_lr[10:20]= alpha/10
    decay_lr[25:35] = lr/10
    decay_lr[35:45]= lr/100
    
    #build model
    if model_type == 'DnCNN':
        model = model_DnCNN.DnCNN()
    elif model_type == 'RIDNet':
        model = DnCNN_Feature_Attention.RIDNet()
    else:
        print(error)
    
    #set up optimizer
    optimizer = tf.optimizers.Adam(learning_rate = lr, epsilon=1e-8, name='AdamOptimizer')

    writer = tf.summary.create_file_writer('./logs/' + name)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer = optimizer, net = model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_directory, max_to_keep=None)
    
    #checkpoint restortion
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        start_epoch = ckpt.save_counter.numpy() + 1
    else:
        print("Initializing from scratch.")
        start_epoch = 1
    
    for epoch in range(start_epoch-1, epochs):
        print('Start of epoch %d' % (epoch,))
        optimizer.learning_rate = decay_lr[epoch]
        train_one_epoch(model, train_dataset, optimizer, writer, ckpt, manager, record_step)
        evaluate_model(model, val_dataset, writer, epoch)
        # save the checkpoint in every epoch
        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(int(epoch), save_path))

    print("Training saved")
    return

if __name__ == '__main__':
    fire.Fire(train_process)

    

   

