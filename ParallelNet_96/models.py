from model_utility import ConvBlock10, WaveletInvLayer, WaveletConvLayer, ConvBlock
from DnCNN_Feature_Attention import EAMBlock
import tensorflow as tf
from tensorflow.keras import layers


class ParallelNet(tf.keras.Model):
    def __init__(self):
        super(ParallelNet, self).__init__()
        self.my_initial = tf.initializers.he_uniform()
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        self.feature_num = 64

        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        
        #self.EAMa = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        #self.EAMb = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        #self.EAMc = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        #self.EAMd = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        
        self. conv10a = ConvBlock10(96, (3,3), self.my_initial, self.my_regular)
        self. conv10b = ConvBlock10(96, (3,3), self.my_initial, self.my_regular)
        self. conv10c = ConvBlock10(96, (3,3), self.my_initial, self.my_regular)
        self. conv10d = ConvBlock10(96, (3,3), self.my_initial, self.my_regular)

        self.conv1 = layers.Conv2D(12, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)

    def call(self, input):
        wav= self.wavelet(input)

        aa = self.conv10a(wav)
        bb = self.conv10b(wav)
        cc = self.conv10c(wav)
        dd = self.conv10d(wav)

        combined = tf.concat([aa, bb, cc, dd], 3)
        after = self.conv1(combined)
        out_ = self.invwavelet(after)
        out = input + out_
        return out


class ParallelNet_5(tf.keras.Model):
    def __init__(self):
        super(ParallelNet_5, self).__init__()
        self.my_initial = tf.initializers.he_uniform()
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)

        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        
        self. conv10a = ConvBlock10(96, (3,3), self.my_initial, self.my_regular)
        self. conv10b = ConvBlock10(96, (3,3), self.my_initial, self.my_regular)
        self. conv10c = ConvBlock10(96, (3,3), self.my_initial, self.my_regular)
        self. conv10d = ConvBlock10(96, (3,3), self.my_initial, self.my_regular)
        self. conv10e = ConvBlock10(96, (3,3), self.my_initial, self.my_regular)

        self.conv1 = layers.Conv2D(12, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)

    def call(self, input):
        wav= self.wavelet(input)

        aa = self.conv10a(wav)
        bb = self.conv10b(wav)
        cc = self.conv10c(wav)
        dd = self.conv10d(wav)
        ee = self.conv10e(wav)
        combined = tf.concat([aa, bb, cc, dd, ee], 3)
        after = self.conv1(combined)
        out_ = self.invwavelet(after)
        out = input + out_
        return out




