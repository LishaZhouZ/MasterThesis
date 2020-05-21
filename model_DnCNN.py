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
    
    self.convlayerEnd = layers.Conv2D(3, (3,3), padding = 'SAME',
        kernel_initializer = self.my_initial,kernel_regularizer = self.my_regular)

    self.convblock1 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)
    self.convblock2 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)
    self.convblock3 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)
    self.convblock4 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)
    self.convblock5 = ConvBlock(64, (3,3), self.my_initial, self.my_regular)

  def call(self, inputs, training=False):
    
    #fist layer
    start = self.convlayerStart(inputs)
    outS = self.reluStart(start)

    out1 = self.convblock1(outS)
    out2 = self.convblock1(out1)
    out3 = self.convblock1(out2)
    out4 = self.convblock1(out3)
    out5 = self.convblock1(out4)

    outE = self.convlayerEnd(out5)
    
    output = outE + inputs
    return output