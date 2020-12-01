import argparse
import glob
from PIL import Image
import PIL
import random
import tensorflow as tf
import time
from pathlib import Path
import os
import numpy as np
parser = argparse.ArgumentParser(description='')
parser.add_argument('--stride', dest='stride', type=int, default=128, help='stride')
parser.add_argument('--step', dest='step', type=int, default = 0, help='escape the first steps')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='DnCNN 64')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help='DnCNN 160, MWCNN 256')
parser.add_argument('--isDebug', dest='isDebug', type=bool, default = False, help='True for 30 images')
parser.add_argument('--save_dir', dest='save_dir', type=str, default = '/mnt/data4/Students/Lisha/patches_128', help='save path')

# check output arguments
args = parser.parse_args()

def generate_patches(dir_label, dir_input, save_dir, tfRecord_name):
    
    filepaths_label = sorted(dir_label.glob('*'))
    
    if args.isDebug:
        numDebug = 4000
        filepaths_label = filepaths_label[:numDebug] # take only ten images to quickly debug
    print("number of training images %d" % len(filepaths_label))
    
    filenames = [item.name[0:-4] + '.jpg' for item in filepaths_label]
    q_input = []
    for qulaity in range(0,101,5):
        q_input.append(Path(dir_input, Path('qp'+str(qulaity))))
    
    count = 0 # calculate the number of patches
    for i in range(len(filepaths_label)):
        img = Image.open(filepaths_label[i])
        im_h, im_w = img.size
        for x in range(0 + args.step, (im_h - args.patch_size), args.stride):
            for y in range(0 + args.step, (im_w - args.patch_size), args.stride):
                count += 1

    origin_patch_num = count
    if origin_patch_num % args.batch_size != 0:
        numPatches = int(origin_patch_num / args.batch_size) * args.batch_size
    else:
        numPatches = int(origin_patch_num)

    print("Total patches = %d , batch size = %d, total batches = %d" %(numPatches, args.batch_size, numPatches / args.batch_size))
    time.sleep(2)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    
    count = 0

    writer = tf.io.TFRecordWriter(save_dir + '/' + tfRecord_name)
    # generate patches
    for i in range(len(filepaths_label)):
        print("The %dth image of %d training images" %(i+1, len(filepaths_label)))
        
        for q in range(0,21):#(13,14): #len(q_input)
            img = Image.open(filepaths_label[i])
            img_input = Image.open(Path(q_input[q], filenames[i]))
            print(q_input[q])
            img_s = np.array(img, dtype="uint8")
            img_s_input = np.array(img_input, dtype="uint8")
            im_h, im_w, im_c = img_s.shape
            for x in range(0 + args.step, im_h - args.patch_size, args.stride):
                for y in range(0 + args.step, im_w - args.patch_size, args.stride):
                    image_label = img_s[x:x + args.patch_size, y:y + args.patch_size, 0:3]
                    image_bayer = img_s_input[x:x + args.patch_size, y:y + args.patch_size, 0:3]
                    image_label = image_label.tobytes()
                    image_bayer = image_bayer.tobytes()
                    count += 1
                    example = tf.train.Example(features = tf.train.Features(feature={
                        'img_label' : tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_label])),
                        'img_bayer' : tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_bayer]))
                    }))
                    if count<= numPatches:
                        writer.write(example.SerializeToString())
                    else:
                        break
    writer.close()
    print("Total patches = %d , batch size = %d, total batches = %d" %(numPatches, args.batch_size, numPatches / args.batch_size))
    print("Data has been written into TFrecord.")

if __name__ == '__main__': 
    src_dir_label = Path("/mnt/data4/Students/Lisha/images/train/groundtruth")
    src_dir_input = Path("/mnt/data4/Students/Lisha/images/train/qp0-100")
    save_dir = args.save_dir
    tfRecord_name = 'train_data_all.tfrecords'
    print("Training data will be generated:")
    generate_patches(src_dir_label, src_dir_input, save_dir, tfRecord_name)

    #For validation data
    #val_dir_label = Path("/mnt/data4/Students/Lisha/images/validation/live1_gt")
    #val_dir_input = Path("/mnt/data4/Students/Lisha/images/validation/live1_0-100")
    #tfRecord_val_name = 'validation_data_q10.tfrecords'
    #print("Validation data will be generated:")
    #generate_patches(val_dir_label, val_dir_input, save_dir, tfRecord_val_name)


