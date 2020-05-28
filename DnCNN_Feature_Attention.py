from model_utility import ConvBlock
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras import layers
from config import channel


batch_size = 64
patch_size = 160
class SoftSchrink(tf.keras.layers.Layer):
  def __init__(self):
    super(SoftSchrink, self).__init__()

  def call(self, x, lower = -0.5, upper = 0.5):
    
    values_below_lower = tf.where(x < lower, x - lower, 0)
    values_above_upper = tf.where(upper < x, x - upper, 0)

    return values_below_lower + values_above_upper

class DnCNN(tf.keras.Model):
  def __init__(self):
    super(DnCNN, self).__init__()
    self.my_initial = tf.initializers.he_normal()
    self.my_regular = tf.keras.regularizers.l2(l=0.0001)
    self.feature_num = 128

    self.convlayerStart = layers.Conv2D(self.feature_num, (3,3), activation='relu', padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
    
    self.convlayerEnd = layers.Conv2D(3, (3,3), activation='relu', padding = 'SAME',
        kernel_initializer = self.my_initial,kernel_regularizer = self.my_regular)

    self.conv1 = layers.Conv2D(self.feature_num, (3,3), activation='relu', padding = 'SAME',
        kernel_initializer=self.my_initial,kernel_regularizer=self.my_regular)# 
    self.conv2 = layers.Conv2D(self.feature_num, (3,3), activation='relu', padding = 'SAME',
        kernel_initializer=self.my_initial,kernel_regularizer=self.my_regular)# 
    self.conv3 = layers.Conv2D(self.feature_num, (3,3), activation='relu', padding = 'SAME',
        kernel_initializer=self.my_initial,kernel_regularizer=self.my_regular)# 
    self.conv4 = layers.Conv2D(self.feature_num, (3,3), activation='relu', padding = 'SAME',
        kernel_initializer=self.my_initial,kernel_regularizer=self.my_regular)# 
    self.conv5 = layers.Conv2D(self.feature_num, (3,3), activation='relu', padding = 'SAME',
        kernel_initializer=self.my_initial,kernel_regularizer=self.my_regular)# 
    self.conv6 = layers.Conv2D(self.feature_num, (3,3), activation='relu', padding = 'SAME',
        kernel_initializer=self.my_initial,kernel_regularizer=self.my_regular)# 
    #self.convblock3 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)
    #self.convblock4 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)
    #self.convblock5 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)

    #the last block there is with 1x1 kernal (in essay denoted by smaller layer)
    self.convlayerRectified = layers.Conv2D(self.feature_num, (1,1), activation='relu', padding = 'SAME',
        kernel_initializer = self.my_initial,kernel_regularizer = self.my_regular)
    
    #feature attention part
    self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
    #downsampling layer with soft-schrinkage funciton activated
    self.convDown = layers.Conv2D(8, (1,1), padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
    self.softschrink = SoftSchrink()

    #upsampling with sigmoid activation
    self.convUp = layers.Conv2D(self.feature_num, (1,1), activation='sigmoid', padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)

    

  def call(self, inputs, training=False):
    #fist layer
    start = self.convlayerStart(inputs)

    out1 = self.conv1(start)
    out2 = self.conv2(out1)
    out2_re = out2 + start

    out3 = self.conv3(out2_re)
    out4 = self.conv4(out3)
    out4_re = out4 + out2_re

    out5 = self.conv5(out4_re)
    out6 = self.conv6(out5)
    out7 = self.convlayerRectified(out6)
    out7_re = out7 + out4_re

    #feature attention block
    F1 = self.global_pooling(out7_re)
    F1 = tf.expand_dims(F1, 1)
    F1 = tf.expand_dims(F1, 1)

    F2 = self.convDown(F1)
    # should be softschrink
    F2_soft = self.softschrink(F2)

    F3 = self.convUp(F2_soft)
    #sigmoid
    
    F4 = F3 * out7_re

    out8 = F4 + start
    
    out9 = self.convlayerEnd(out8)
    
    output = out9 + inputs
    
    return output
