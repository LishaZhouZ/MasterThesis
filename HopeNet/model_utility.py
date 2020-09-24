import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers

#from utils_py3_tfrecord_2 import *

#return the average psnr for
#PSNR Metric
class PSNRMetric(tf.keras.metrics.Metric):
  def __init__(self, name='psnr', **kwargs):
    super(PSNRMetric, self).__init__(name=name, **kwargs)
    self.psnr = self.add_weight(name='psnr', initializer='zeros')
    self.count = self.add_weight(name='count', initializer='zeros')
  
  def update_state(self, y_true, y_pred):
    psnr1 = tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=255.0))
    self.psnr.assign_add(psnr1)
    self.count.assign_add(1)

  def result(self):
    return self.psnr/self.count

#MS_SSIMMetric
class MS_SSIMMetric(tf.keras.metrics.Metric):
  def __init__(self, name='psnr', **kwargs):
    super(MS_SSIMMetric, self).__init__(name=name, **kwargs)
    self.ms_ssim = self.add_weight(name='ms_ssim', initializer='zeros')
    self.count = self.add_weight(name='count', initializer='zeros')
  
  def update_state(self, y_true, y_pred):
    mssim = tf.reduce_mean(tf.image.ssim(y_pred, y_true, 255.0))
    self.ms_ssim.assign_add(mssim) #output is 4x1 array
    self.count.assign_add(1)

  def result(self):
    return self.ms_ssim/self.count

#l2 loss
def loss_l2(prediction, groundtruth):
  #inv_converted = wavelet_inverse_conversion(prediction)
  frobenius_norm = tf.norm(prediction-groundtruth, ord='fro', axis=(1, 2))
  lossRGB = (1/2)*(tf.reduce_mean(frobenius_norm**2))
  #regularization loss
  return lossRGB

#l2 loss
def loss_l1(prediction, groundtruth):
  #inv_converted = wavelet_inverse_conversion(prediction)
  absLoss = tf.abs(prediction-groundtruth)
  lossSum = tf.reduce_sum(absLoss,[1,2,3])
  lossRGB = tf.reduce_mean(lossSum)
  #regularization loss
  return lossRGB

#Wavelet layer
class WaveletConvLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(WaveletConvLayer, self).__init__()

  def call(self, inputs):
    inputs = inputs/4
    im_c1 = inputs[:, 0::2, 0::2, :] # 1
    im_c2 = inputs[:, 0::2, 1::2, :] # right up
    im_c3 = inputs[:, 1::2, 0::2, :] # left down
    im_c4 = inputs[:, 1::2, 1::2, :] # right right

    LL = im_c1 + im_c2 + im_c3 + im_c4
    LH = -im_c1 - im_c2 + im_c3 + im_c4
    HL = -im_c1 + im_c2 - im_c3 + im_c4
    HH = im_c1 - im_c2 - im_c3 + im_c4
    result = tf.concat([LL, LH, HL, HH], 3) #(None, 96,96,12)    
    return result

class WaveletInvLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(WaveletInvLayer, self).__init__()

  def call(self, inputs):
    sz = inputs.shape
    inputs = inputs
    a = tf.cast(sz[3]/4, tf.int32)
    LL = inputs[:, :, :, 0:a]
    LH = inputs[:, :, :, a:2*a]
    HL = inputs[:, :, :, 2*a:3*a]
    HH = inputs[:, :, :, 3*a:]
    
    aa = LL - LH - HL + HH
    bb = LL - LH + HL - HH
    cc = LL + LH - HL - HH
    dd = LL + LH + HL + HH
    concated = tf.concat([aa, bb, cc, dd], 3)
    reconstructed = tf.nn.depth_to_space(concated, 2)
    return reconstructed

class ConvConcatLayer(layers.Layer):
  def __init__(self, feature_num, kernel_size, my_initial, my_regular, dilated=1):
    super(ConvConcatLayer, self).__init__()
    self.conv = layers.Conv2D(feature_num, kernel_size, dilation_rate = dilated, padding = 'SAME',
        kernel_initializer=my_initial,kernel_regularizer=my_regular)# 
    self.bn = layers.BatchNormalization()
    self.relu = layers.ReLU()
  
  def call(self, inputs):
    a = self.conv(inputs)
    b = self.bn(a)
    c = self.relu(b)
    return c

class ConvBlock(layers.Layer):
  def __init__(self, feature_num, kernel_size, my_initial, my_regular):
    super(ConvBlock, self).__init__()
    self.alpha1 = ConvConcatLayer(feature_num, kernel_size, my_initial, my_regular)
    self.alpha2 = ConvConcatLayer(feature_num, kernel_size, my_initial, my_regular)
    self.alpha3 = ConvConcatLayer(feature_num, kernel_size, my_initial, my_regular)
    self.alpha4 = ConvConcatLayer(feature_num, kernel_size, my_initial, my_regular)
  
  def call(self, inputs):
    a11 = self.alpha1(inputs)
    a12 = self.alpha2(a11)
    a13 = self.alpha3(a12)
    a14 = self.alpha4(a13)
    return a14

class ConvInvBlock(layers.Layer):
  def __init__(self, feature_num1, kernel_size, my_initial, my_regular):
    super(ConvInvBlock, self).__init__()
    self.alpha1 = ConvConcatLayer(feature_num1, kernel_size, my_initial, my_regular)
    self.alpha2 = ConvConcatLayer(feature_num1, kernel_size, my_initial, my_regular)
    self.alpha3 = ConvConcatLayer(feature_num1, kernel_size, my_initial, my_regular)
  
  def call(self, inputs):
    a11 = self.alpha1(inputs)
    a12 = self.alpha2(a11)
    a13 = self.alpha3(a12)
    return a13

class HopeNet(tf.keras.Model):
  def __init__(self):
    super(HopeNet, self).__init__()
    self.my_initial = tf.initializers.he_normal()
    self.my_regular = tf.keras.regularizers.l2(l=0.0001)

    self.convblock_LL = ConvBlock(120, (3,3), self.my_initial, self.my_regular)
    self.convblock_HL = ConvBlock(120, (3,3), self.my_initial, self.my_regular)
    self.convblock_LH = ConvBlock(120, (3,3), self.my_initial, self.my_regular)
    self.convblock_HH = ConvBlock(120, (3,3), self.my_initial, self.my_regular)
    
    self.invConvblock_LL = ConvBlock(3, (3,3), self.my_initial, self.my_regular)
    self.invConvblock_HL = ConvBlock(3, (3,3), self.my_initial, self.my_regular)
    self.invConvblock_LH = ConvBlock(3, (3,3), self.my_initial, self.my_regular)
    self.invConvblock_HH = ConvBlock(3, (3,3), self.my_initial, self.my_regular)

    #wavelet part
    self.wavelet = WaveletConvLayer()

    self.invwavelet = WaveletInvLayer()

  
  def call(self, inputs):

    wav = self.wavelet(inputs)  #
    # seperate to LL and HL,LH,HH part
    LL = wav[:, :, :, 0:3]
    LH = wav[:, :, :, 3:6]
    HL = wav[:, :, :, 6:9]
    HH = wav[:, :, :, 9:12]

    #with LL only apply one 4-FCN layers
    conLL = self.convblock_LL(LL)
    invConLL = self.invConvblock_LL(conLL)

    conLH = self.convblock_LH(LH)
    invConLH = self.invConvblock_LH(conLH)

    conHL = self.convblock_HL(HL)
    invConHL = self.invConvblock_HL(conHL)
    
    conHH = self.convblock_HH(HH)
    invConHH = self.invConvblock_HH(conHH)

    resultInter = tf.concat([invConLL, invConLH, invConHL, invConHH], 3)
    invwav = self.invwavelet(resultInter)  #1024-256
    
    output = invwav + inputs
    return output
    

