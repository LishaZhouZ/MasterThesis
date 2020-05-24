from model_utility import ConvBlock
import tensorflow as tf
from tensorflow.keras import layers

batch_size = 64
patch_size = 160

class DnCNN(tf.keras.Model):
  def __init__(self):
    super(DnCNN, self).__init__()
    self.my_initial = tf.initializers.he_normal()
    self.my_regular = tf.keras.regularizers.l2(l=0.0001)
    

    self.convlayerStart = layers.Conv2D(64, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
    self.reluStart = layers.ReLU()
    
    self.convlayerEnd = layers.Conv2D(12, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial,kernel_regularizer = self.my_regular)

    self.convblock1 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)
    self.convblock2 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)
    self.convblock3 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)
    self.convblock4 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)
    self.convblock5 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)

  def call(self, inputs, training=False):
    input_arranged = tf.nn.space_to_depth(inputs, 2, data_format='NHWC', name=None)
    #fist layer
    start = self.convlayerStart(input_arranged)
    outS = self.reluStart(start)

    out1 = self.convblock1(outS)
    out2 = self.convblock2(out1)
    out3 = self.convblock3(out2)
    out4 = self.convblock4(out3)
    out5 = self.convblock5(out4)

    outE = self.convlayerEnd(out5)

    output_rearranged = tf.nn.depth_to_space(outE, 2, data_format='NHWC', name=None)

    output = output_rearranged + inputs
    return output
