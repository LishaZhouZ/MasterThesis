# this file restore the model and checkpoint from file, and to check resulting image
from seq_model_MWCNN import *
import os
from pathlib import Path
import matplotlib.pyplot as plt
import glob
from PIL import Image
import numpy as np
import math

checkpoint_directory = './tf_ckpts'

def image_cutting(dir_label, dir_input):
    filepaths_label = sorted(dir_label.glob('*'))
    filepaths_input = sorted(dir_input.glob('*'))
    count = 0
    for i in range(len(filepaths_label)):
        img_label = Image.open(filepaths_label[i])
        img_input = Image.open(filepaths_input[i])
        
        img_s_label = np.array(img_label, dtype="uint8")
        img_s_input = np.array(img_input, dtype="uint8")

        im_h, im_w, im_c = img_s_label.shape
        print("The %dth image of %d training images" %(i+1, len(filepaths_label)))
        for x in range(0, im_h - patch_size, patch_size):
            for y in range(0, im_w - patch_size, patch_size):
                image_label = img_s_label[x:x + patch_size, y:y + patch_size, 0:3] # some images have an extra blank channel 
                image_bayer = img_s_input[x:x + patch_size, y:y + patch_size, 0:3]
                
                result_label = Image.fromarray((image_label).astype(np.uint8))
                result_label.save('./images/test/split_label/' + str(count) + '.bmp')

                result_input = Image.fromarray((image_bayer).astype(np.uint8))
                result_input.save('./images/test/split_input/' + str(count) + '.bmp')
                count += 1

def feature_visualization(model, img):
    model_modified = model(inputs=model.inputs, outputs=model.layers[1].output)
    # get feature map for first hidden layer
    feature_maps = model_modified.predict(img)
    # plot all 64 maps in an 8x8 squares
    square = 8
    ix = 1
    for _ in range(square):
    	for _ in range(square):
    		# specify subplot and turn of axis
    		ax = plt.subplot(square, square, ix)
    		ax.set_xticks([])
    		ax.set_yticks([])
    		# plot filter channel in grayscale
    		plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
    		ix += 1
    # show the figure
    plt.show()

def TrainingSetTF():
    train_dataset = read_and_decode('./patches/MWCNN_train_data.tfrecords')
    img_input,img_label = next(iter(train_dataset))
    img_s_input = img_input[1,:,:,:]
    img_s_label = img_label[1,:,:,:]

    return img_s_input, img_s_label

def ImageFromPath(number):
    #image_cutting(Path('./images/test/live1_groundtruth'), Path('./images/test/live1_qp10'))
    #filepaths_label = sorted(Path('./images/test/split_label').glob('*'))
    #filepaths_input = sorted(Path('./images/test/split_input').glob('*'))
    path_label = Path('/home/lisha/Forschungspraxis/images/test/split_label/' + str(number) + '.bmp')
    path_input = Path('/home/lisha/Forschungspraxis/images/test/split_input/' + str(number) + '.bmp')

    img_label = Image.open(path_label)
    img_input = Image.open(path_input)
    img_s_label = np.array(img_label, dtype="float32")
    img_s_input = np.array(img_input, dtype="float32")
    
    return img_s_input,img_s_label

if __name__ == "__main__":
    dir_label = Path('./images/test/live1_groundtruth/buildings.bmp')
    dir_input = Path('./images/test/live1_qp10/buildings.jpg')

    mode = 3 #--can be 1,2,3
    #model 1---------------
    if mode==1:
        model = MWCNN()
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = model)
        ckpt.restore(tf.train.latest_checkpoint(Path('./ckpt'))).expect_partial()
    elif mode==2:
    #model 2-----------------
        model = MWCNN_m1()
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = model)
        ckpt.restore(tf.train.latest_checkpoint(Path('./ckpt-m1'))).expect_partial()
    elif mode ==3:
    #model 2---------------
        model = MWCNN_m2()
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), net = model)
        ckpt.restore(tf.train.latest_checkpoint(Path('./ckpt-m2'))).expect_partial()
    else:
        print("error")

    img_label = Image.open(dir_label)
    img_input = Image.open(dir_input)
     
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
    
    
    output = model(img_s_input_batch, training =False)
    reconstructed = img_s_input_batch + output
    output_cut = tf.slice(reconstructed, [0, padding_up, padding_left, 0], [1, shape_input[0], shape_input[1], 3])
    
    org_psnr = tf.image.psnr(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0).numpy()
    rec_psnr = tf.image.psnr(output_cut, img_s_label_batch, 255.0).numpy()
    org_ssim= tf.image.ssim_multiscale(img_s_label_batch, tf.expand_dims(img_s_input, axis = 0), 255.0)
    rec_ssim = tf.image.ssim_multiscale(output_cut, img_s_label_batch, 255.0)
    print('Image ' + ' org_psnr:%.4f,' % org_psnr + 'after_psnr:%.4f,' % rec_psnr, ' org_ssim:%.4f,' % org_ssim + 'after_ssim:%.4f' % rec_ssim)
        
    save_images(Path('/home/lisha/Forschungspraxis/images/test/outcome/' + str(mode) + '.bmp'), img_s_label, noisy_image = img_s_input.numpy(), clean_image = np.squeeze(output_cut, axis=0))
    
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