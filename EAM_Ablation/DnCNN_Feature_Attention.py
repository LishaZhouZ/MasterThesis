from model_utility import ConvBlock, ConvConcatLayer, WaveletConvLayer, WaveletInvLayer
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras import layers
import numpy as np


batch_size = 64
patch_size = 80
class SoftSchrink(tf.keras.layers.Layer):
    def __init__(self):
        super(SoftSchrink, self).__init__()
    
    def call(self, x, lower = -0.0001, upper = 0.0001):
        values_below_lower = tf.where(x < lower, x - lower, 0)
        values_above_upper = tf.where(upper < x, x - upper, 0)
        return values_below_lower + values_above_upper

class MeanShift(tf.keras.layers.Conv2D):
    def __init__(
        self, rgb_range, rgb_mean, sign=-1, trainable = False,
        kernel_initializer = tf.constant_initializer(value=1), use_bias = True):
        
        super(MeanShift, self).__init__(3, kernel_size = 1)
        self.bias = tf.convert_to_tensor(sign * rgb_range * rgb_mean)


class FeatureAttention(layers.Layer):
    def __init__(self, input_channel, my_initial, my_regular):
        super(FeatureAttention, self).__init__()
        #downsampling layer with soft-schrinkage funciton activated
        self.convDown = layers.Conv2D(input_channel//16, (1,1), activation = 'relu', padding = 'VALID',
            kernel_initializer = my_initial, kernel_regularizer = my_regular)
        #self.softschrink = SoftSchrink()
         
        #upsampling with sigmoid activation
        self.convUp = layers.Conv2D(input_channel, (1,1), activation='sigmoid', padding = 'VALID',
            kernel_initializer = my_initial, kernel_regularizer = my_regular)
    
    def call(self, input):
        F1 = tf.nn.avg_pool2d(input, [input.shape[1], input.shape[2]], 1, 'VALID', data_format='NHWC', name=None)
        F2 = self.convDown(F1)
        F3 = self.convUp(F2)
        return input*F3

class MergeAndRun(layers.Layer):
    def __init__(self, feature_num, my_initial, my_regular):
        super(MergeAndRun, self).__init__()
        #first branch
        self.conv11= layers.Conv2D(feature_num, (3,3), dilation_rate = 1, activation = 'relu', padding = 'SAME',
           kernel_initializer = my_initial, kernel_regularizer = my_regular)# 
        self.conv12 = layers.Conv2D(feature_num, (3,3), dilation_rate = 2, activation = 'relu', padding = 'SAME',
           kernel_initializer = my_initial, kernel_regularizer = my_regular)# 
        
        #second branch
        self.conv21= layers.Conv2D(feature_num, (3,3), dilation_rate = 3, activation = 'relu', padding = 'SAME',
           kernel_initializer = my_initial, kernel_regularizer = my_regular)# 
        self.conv22 = layers.Conv2D(feature_num, (3,3), dilation_rate = 4, activation = 'relu', padding = 'SAME',
           kernel_initializer = my_initial, kernel_regularizer = my_regular)# 
        
        self.convCombined = layers.Conv2D(feature_num, (3,3), activation = 'relu', padding = 'SAME',
           kernel_initializer = my_initial, kernel_regularizer = my_regular)# 
    
    def call(self, input):
        #first branch
        R1 = self.conv11(input)
        out1 = self.conv12(R1)
        
        #second branch
        R2 = self.conv21(input)
        out2 = self.conv22(R2)

        Merge = tf.concat([out1, out2], 3)
        out3 = self.convCombined(Merge)

        #changes only for check inception model
        out = out3 + input
        
        return out

class ResidualBlock(layers.Layer):
    def __init__(self, feature_num, my_initial, my_regular):
        super(ResidualBlock, self).__init__()
        self.conv1= layers.Conv2D(feature_num, (3,3), activation = 'relu', padding = 'SAME',
           kernel_initializer = my_initial, kernel_regularizer = my_regular)# 
        self.conv2 = layers.Conv2D(feature_num, (3,3), padding = 'SAME',
           kernel_initializer = my_initial, kernel_regularizer = my_regular)# 
        
    def call(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        # add together first then relu
        out3 = tf.nn.relu(out2 + input)
        return out3

class EResidualBlock(layers.Layer):
    def __init__(self, feature_num, my_initial, my_regular):
        super(EResidualBlock, self).__init__()
        self.conv1= layers.Conv2D(feature_num, (3,3), activation = 'relu', padding = 'SAME',
           kernel_initializer = my_initial, kernel_regularizer = my_regular)# 
        self.conv2= layers.Conv2D(feature_num, (3,3), activation = 'relu', padding = 'SAME',
           kernel_initializer = my_initial, kernel_regularizer = my_regular)# 

        self.conv3 = layers.Conv2D(feature_num, (1,1), padding = 'VALID',
           kernel_initializer = my_initial, kernel_regularizer = my_regular)# 
        
    def call(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        # add together first then relu
        out4 = tf.nn.relu(out3 + input)

        return out4

class EAMBlock(layers.Layer):
    def __init__(self, feature_num, my_initial, my_regular):
        super(EAMBlock, self).__init__()
        self.r1 = MergeAndRun(feature_num, my_initial, my_regular)
        self.r2 = ResidualBlock(feature_num, my_initial, my_regular)
        self.r3 = EResidualBlock(feature_num, my_initial, my_regular)
        #self.fe = FeatureAttention(feature_num, my_initial, my_regular)
    
    def call(self, input):
        r1 = self.r1(input)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        #out = self.fe(r3)
        return r3



class RIDNet(tf.keras.Model):
  def __init__(self):
    super(RIDNet, self).__init__()
    self.my_initial = tf.initializers.glorot_uniform()
    self.my_regular = tf.keras.regularizers.l2(l=0.0001)
    self.feature_num = 64

    self.head = layers.Conv2D(self.feature_num, (3,3), activation = 'relu', padding = 'SAME',
            kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)# 
    self.tail = layers.Conv2D(3, (3,3), padding = 'SAME',
            kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)# 

    self.EAM1 = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
    self.EAM2 = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
    self.EAM3 = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
    self.EAM4 = EAMBlock(self.feature_num, self.my_initial, self.my_regular)

    self.rgb_mean = np.array([0.4488, 0.4371, 0.4040])
    self.rgb_std = np.array([1.0, 1.0, 1.0])
    self.rgb_range = 255.0

    self.sub_mean = MeanShift(self.rgb_range, self.rgb_mean, self.rgb_std)       
    self.add_mean = MeanShift(self.rgb_range, self.rgb_mean, self.rgb_std, 1)

  def call(self, input, training=False):
    #sub mean
    pre = self.sub_mean(input)
    #fist layer
    start = self.head(pre)
    
    eam1 = self.EAM1(start)
    eam2 = self.EAM2(eam1)
    eam3 = self.EAM3(eam2)
    eamOut = self.EAM4(eam3)
    
    res = self.tail(eamOut)
    #add mean
    post = self.add_mean(res)

    out = post + input

    return out

class AWNet(tf.keras.Model):
  def __init__(self):
    super(AWNet, self).__init__()
    self.my_initial = tf.initializers.glorot_normal()
    self.my_regular = tf.keras.regularizers.l2(l=1)
    self.feature_num = 64

    self.head = layers.Conv2D(self.feature_num, (3,3), activation = 'relu', padding = 'SAME',
            kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)# 
    self.tail = layers.Conv2D(3, (3,3), padding = 'SAME',
            kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)# 

    self.EAM1 = EAMBlock(256, self.my_initial, self.my_regular)
    self.EAM2 = EAMBlock(1024, self.my_initial, self.my_regular)
    self.EAM3 = EAMBlock(256, self.my_initial, self.my_regular)
    
    self.EAM4 = EAMBlock(256, self.my_initial, self.my_regular)
    self.EAM5 = EAMBlock(64, self.my_initial, self.my_regular)
    
    #wavelet layer
    self.wavelet1 = WaveletConvLayer()
    self.wavelet2 = WaveletConvLayer()
    self.wavelet3 = WaveletConvLayer()
    
    self.invwavelet1 = WaveletInvLayer()
    self.invwavelet2 = WaveletInvLayer()
    self.invwavelet3 = WaveletInvLayer()
    self.convlayer1024 = layers.Conv2D(1024, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
    self.convlayer640 = layers.Conv2D(640, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial,kernel_regularizer = self.my_regular)
    self.convlayer64 = layers.Conv2D(64, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)

    self.rgb_mean = np.array([0.4488, 0.4371, 0.4040])
    self.rgb_std = np.array([1.0, 1.0, 1.0])
    self.rgb_range = 255.0

    self.sub_mean = MeanShift(self.rgb_range, self.rgb_mean, self.rgb_std)       
    self.add_mean = MeanShift(self.rgb_range, self.rgb_mean, self.rgb_std, 1)

  def call(self, input, training=False):
    #sub mean
    pre = self.sub_mean(input)
    #fist layer
    start = self.head(pre)      #3-64
    wav1 = self.wavelet1(start)  #64-256
    eam1 = self.EAM1(wav1)       #256 - 256

    wav2 = self.wavelet2(eam1)   #256-1024
    eam2 = self.EAM2(wav2)      #1024-1024
    
#    wav3 = self.wavelet3(eam2)  #256-1024
#    eam3 = self.EAM3(wav3)      #1024-256
#    eam3_expand = self.convlayer1024(eam3) #-1024
    
    invwav3 = self.invwavelet3(eam2) #1024-256
    eam4 = self.EAM4(invwav3+eam1)  #256-256
    #eam4_expand = self.convlayer640(eam4) #256-640

    invwav2 = self.invwavelet2(eam4) #256-64
    eam5 = self.EAM5(invwav2 + start) #64

#    eam5_rectified = self.convlayer64(eam5) #160-64
    #invwav1 = self.invwavelet1(eam5)

    res = self.tail(eam5)
    #add mean
    post = self.add_mean(res)
    return post


class AWNet2(tf.keras.Model):
  def __init__(self):
    super(AWNet2, self).__init__()
    self.my_initial = tf.initializers.glorot_normal()
    self.my_regular = tf.keras.regularizers.l2(l=0.0001)
    self.feature_num = 64

    self.head = layers.Conv2D(self.feature_num, (3,3), activation = 'relu', padding = 'SAME',
            kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)# 
    self.tail = layers.Conv2D(3, (3,3), padding = 'SAME',
            kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)# 

    self.EAM1 = EAMBlock(256, self.my_initial, self.my_regular)
    self.EAM2 = EAMBlock(1024, self.my_initial, self.my_regular)
    self.EAM3 = EAMBlock(256, self.my_initial, self.my_regular)
    
    self.EAM4 = EAMBlock(256, self.my_initial, self.my_regular)
    self.EAM5 = EAMBlock(64, self.my_initial, self.my_regular)
    
    #wavelet layer
    self.wavelet1 = WaveletConvLayer()
    self.wavelet2 = WaveletConvLayer()
    self.wavelet3 = WaveletConvLayer()
    
    self.invwavelet1 = WaveletInvLayer()
    self.invwavelet2 = WaveletInvLayer()
    self.invwavelet3 = WaveletInvLayer()
    self.convlayer1024 = layers.Conv2D(1024, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
    self.convlayer640 = layers.Conv2D(640, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial,kernel_regularizer = self.my_regular)
    self.convlayer64 = layers.Conv2D(64, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)

    self.rgb_mean = np.array([0.4488, 0.4371, 0.4040])
    self.rgb_std = np.array([1.0, 1.0, 1.0])
    self.rgb_range = 255.0

    self.sub_mean = MeanShift(self.rgb_range, self.rgb_mean, self.rgb_std)       
    self.add_mean = MeanShift(self.rgb_range, self.rgb_mean, self.rgb_std, 1)

  def call(self, input, training=False):
    #sub mean
    pre = self.sub_mean(input)
    #fist layer
    start = self.head(pre)      #3-64
    wav1 = self.wavelet1(start)  #64-256
    eam1 = self.EAM1(wav1)       #256 - 256

    wav2 = self.wavelet2(eam1)   #256-1024
    eam2 = self.EAM2(wav2)      #1024-1024
    
    wav3 = self.wavelet3(eam2)  #1024- 4096
    eam3 = self.EAM3(wav3)      #4096-4096
#    eam3_expand = self.convlayer1024(eam3) #-1024
    

    invwav3 = self.invwavelet3(eam3) #1024-256
    eam4 = self.EAM4(invwav3+eam2)  #256-256
    #eam4_expand = self.convlayer640(eam4) #256-640

    invwav2 = self.invwavelet2(eam4) #256-64
    eam5 = self.EAM5(invwav2 + eam1) #64

#    eam5_rectified = self.convlayer64(eam5) #160-64
    invwav1 = self.invwavelet1(eam5)
    

    res = self.tail(invwav1 + start)
    #add mean
    post = self.add_mean(res)
    return post