import sys
#sys.path.insert(0, '/home/ge29nab/MasterThesis')
import argparse
from glob import glob
import datetime
import os
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import math
from utils_py3_tfrecord_128 import read_and_decode
from train_model import train_one_epoch, evaluate_model
import models
import fire
import numpy as np
#weigth decay momentum optimizer
#L2 regularization
#tensorboard

def train_process(train_dataset_path = '/mnt/data4/Students/Lisha/patches/train_data_q10_128.tfrecords', 
    lr = 0.01, ckpt_dir = '/mnt/data4/Students/Lisha/tf_ckpts/', name='TestNet_waveletLoss', batch_size = 32, epochs = 40):
    gpus = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass
    train_dataset = read_and_decode(train_dataset_path, batch_size)

    record_step = 1
    ckpt_directory = ckpt_dir + name
    decay_lr = np.ones(epochs+1)
    decay_lr[0:20]= lr
    decay_lr[20:30]= lr/10
    decay_lr[30:41] = lr/100
    
    model1 = models.TestNet_LL()
    model2 = models.TestNet_LLHH()    

    #set up optimizer
    optimizer = tf.optimizers.Adam(learning_rate = lr, epsilon=1e-8, name='AdamOptimizer')

    logdir = '/home/ge29nab/MasterThesis/logs/' + name
    ckpt1 = tf.train.Checkpoint(step=tf.Variable(1), optimizer = optimizer, net = model1)
    ckpt2 = tf.train.Checkpoint(step=tf.Variable(1), optimizer = optimizer, net = model2)
    
    manager1 = tf.train.CheckpointManager(ckpt1, ckpt_directory+'model1', max_to_keep=None)
    manager2 = tf.train.CheckpointManager(ckpt2, ckpt_directory+'model2', max_to_keep=None)
    #checkpoint restortion
    ckpt1.restore(manager1.latest_checkpoint)
    ckpt2.restore(manager2.latest_checkpoint)
    if manager1.latest_checkpoint:
        print("Restored from {}".format(manager1.latest_checkpoint))
        start_epoch = ckpt1.save_counter.numpy() + 1
    else:
        print("Initializing from scratch.")
        start_epoch = 1
    
    for epoch in range(start_epoch-1, epochs):
        print('Start of epoch %d' % (epoch,))
        optimizer.learning_rate = decay_lr[epoch]
        train_one_epoch(model1, model2, train_dataset, optimizer, logdir, ckpt1, manager1, record_step)
        evaluate_model(model1, model2, logdir, epoch)
        # save the checkpoint in every epoch
        save_path = manager1.save()
        print("Saved checkpoint for epoch {}: {}".format(int(epoch), save_path))
        
    print("Training saved")
    return

if __name__ == '__main__':
    fire.Fire(train_process)

    

   

