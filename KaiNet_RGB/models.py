from model_utility import ConvBlock10, WaveletInvLayer, WaveletConvLayer, ConvBlock
from DnCNN_Feature_Attention import EAMBlock
import tensorflow as tf
from tensorflow.keras import layers

class KaiNet_RGB(tf.keras.Model):
    def __init__(self):
        super(KaiNet_RGB, self).__init__()
        self.my_initial = tf.initializers.he_normal()
        
        self.my_regular = tf.keras.regularizers.l2(l=0.0005)
        self.feature_num = 64

        self. conv10R = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10G = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10B = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
    
        self.convR = layers.Conv2D(1, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.convG = layers.Conv2D(1, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.convB = layers.Conv2D(1, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        

    def call(self, input):
        RR = input[:, :, :, 0:1]
        GG = input[:, :, :, 1:2]
        BB = input[:, :, :, 2:3]

        afterGG = self.conv10G(GG)
        afterG= self.convG(afterGG)

        concatedRG = tf.concat([RR, afterG], 3)
        afterRG = self.conv10R(concatedRG)
        afterR = self.convR(afterRG)

        concatedGB = tf.concat([afterG, BB], 3) 
        afterGB = self.conv10B(concatedGB)
        afterB = self.convB(afterGB)
        
        concatedRGB = tf.concat([afterR, afterG, afterB], 3) 
        out = input + concatedRGB
        return out

