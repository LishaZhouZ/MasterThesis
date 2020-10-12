from model_utility import ConvBlock10, WaveletInvLayer, WaveletConvLayer, ConvBlock
import tensorflow as tf
from tensorflow.keras import layers
class TestNet(tf.keras.Model):
    def __init__(self):
        super(TestNet, self).__init__()
        self.my_initial = tf.initializers.he_normal()
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        self.feature_num = 64

        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        self. conv10a = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10b = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10c = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10d = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)

        self. conv_after = ConvBlock(64, (3,3), self.my_initial, self.my_regular)
        self.tail = layers.Conv2D(12, (3,3), padding = 'SAME',
                kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)# 

    def call(self, input):
        wav= self.wavelet(input)
        LL = wav[:, :, :, 0:3]
        LH = wav[:, :, :, 3:6]
        HL = wav[:, :, :, 6:9]
        HH = wav[:, :, :, 9:12]

        LL_conv = self.conv10a(wav)
        LH_conv = self.conv10b(wav)
        HL_conv = self.conv10c(wav)
        HH_conv = self.conv10d(wav)

        post_LL = LL + LL_conv
        post_LH = LH + LH_conv
        post_HL = HL + HL_conv
        post_HH = HH + HH_conv

        combined = tf.concat([post_LL, post_LH, post_HL, post_HH], 3)
        out = self.invwavelet(combined)

        return out