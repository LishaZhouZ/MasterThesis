import sys
sys.path.insert(0, '/home/ge29nab/MasterThesis')
import argparse
from glob import glob
import datetime
import os
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import math
from utils_py3_tfrecord_256 import read_and_decode
from train_model_l1 import train_one_epoch, evaluate_model
import DnCNN_Feature_Attention
import fire
import numpy as np

#weigth decay momentum optimizer
#L2 regularization
#tensorboard

def train_process(train_dataset_path = '/mnt/data4/Students/Lisha/patches_256/train_data_256.tfrecords', 
    val_dataset_path = '/mnt/data4/Students/Lisha/patches_256/validation_data_256.tfrecords', 
    lr = 0.001, ckpt_dir = '/mnt/data4/Students/Lisha/tf_ckpts/', name='AWNet-all', model_type = 'AWNet', batch_size =4 , epochs = 70):
    gpus = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        #tf.config.experimental.set_memory_growth(gpus[1], True)
    except:
        pass
    train_dataset = read_and_decode(train_dataset_path, batch_size)
    val_dataset = read_and_decode(val_dataset_path, batch_size)
    record_step = 10
    ckpt_directory = ckpt_dir + name
    decay_lr = np.ones(epochs+1)
    decay_lr[0:10]= lr
    decay_lr[10:20]= lr/10
    decay_lr[20:30] = lr/100
    decay_lr[30:40]= lr/1000
    decay_lr[40:50]= lr/10000
    decay_lr[50:60]= lr/50000
    decay_lr[60:70]= lr/100000
    #build model
    model = DnCNN_Feature_Attention.AWNet()
    
    #set up optimizer
    optimizer = tf.optimizers.Adam(learning_rate = lr, epsilon=1e-8, name='AdamOptimizer')

    writer = tf.summary.create_file_writer('/home/ge29nab/MasterThesis/logs/' + name)
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

    

   

