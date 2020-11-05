from model_utility import ConvBlock10, WaveletInvLayer, WaveletConvLayer, ConvBlock
from DnCNN_Feature_Attention import EAMBlock
import tensorflow as tf
from tensorflow.keras import layers


class TestNet4(tf.keras.Model):
    def __init__(self):
        super(TestNet4, self).__init__()
        self.my_initial = tf.initializers.glorot_uniform()
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        self.feature_num = 64

        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        
        self.EAMa = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        self.EAMb = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        self.EAMc = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        self.EAMd = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        
        #self. conv10a = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        #self. conv10b = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        #self. conv10c = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        #self. conv10d = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)

        self.conv1 = layers.Conv2D(self.feature_num, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv2 = layers.Conv2D(self.feature_num, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv3 = layers.Conv2D(self.feature_num, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv4 = layers.Conv2D(self.feature_num, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.convAF = layers.Conv2D(12, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        
        self.conv5 = layers.Conv2D(3, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv6 = layers.Conv2D(3, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv7 = layers.Conv2D(3, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv8 = layers.Conv2D(3, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)


    def call(self, input):
        wav= self.wavelet(input)
        a1 = self.conv1(wav)
        b1 = self.conv2(wav)
        c1 = self.conv3(wav)
        d1 = self.conv4(wav)
        
        aa = self.EAMa(a1)
        bb = self.EAMb(b1)
        cc = self.EAMc(c1)
        dd = self.EAMd(d1)

        aa1 = self.conv5(aa)
        bb1 = self.conv6(bb)
        cc1 = self.conv7(cc)
        dd1 = self.conv8(dd)

        combined = tf.concat([aa1, bb1, cc1, dd1], 3)
        after = self.convAF(combined)
        out_ = self.invwavelet(after)
        out = input + out_
        return out