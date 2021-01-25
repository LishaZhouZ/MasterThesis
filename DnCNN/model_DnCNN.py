from model_utility import ConvBlock10
import tensorflow as tf
from tensorflow.keras import layers

batch_size = 64


class DnCNN(tf.keras.Model):
  def __init__(self):
    super(DnCNN, self).__init__()
    self.my_initial = tf.initializers.he_normal()
    self.my_regular = tf.keras.regularizers.l2(l=0.0001)
    

    self.convlayerStart = layers.Conv2D(64, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
    self.reluStart = layers.ReLU()
    
    self.convlayerEnd = layers.Conv2D(3, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial,kernel_regularizer = self.my_regular)

    self.convblock1 = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
    

  def call(self, inputs, training=False):
    #input_arranged = tf.nn.space_to_depth(inputs, 2, data_format='NHWC', name=None)
    #fist layer
    start = self.convlayerStart(inputs)
    outS = self.reluStart(start)

    out1 = self.convblock1(outS)

    outE = self.convlayerEnd(out1)

    #output_rearranged = tf.nn.depth_to_space(outE, 2, data_format='NHWC', name=None)

    output = outE + inputs
    return output
