from model_utility import ConvBlock10, WaveletInvLayer, WaveletConvLayer, ConvBlock
from DnCNN_Feature_Attention import EAMBlock
import tensorflow as tf
from tensorflow.keras import layers

#best with pz 128, 27.29
class TestNet(tf.keras.Model):
    def __init__(self):
        super(TestNet, self).__init__()
        self.my_initial = tf.initializers.he_uniform()
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        self.feature_num = 64

        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        self. conv10a = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10b = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10c = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10d = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)

        self.conv1 = layers.Conv2D(3, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv2 = layers.Conv2D(3, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv3 = layers.Conv2D(3, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv4 = layers.Conv2D(3, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)

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
        
        LL_convf = self.conv1(LL_conv)
        LH_convf = self.conv2(LH_conv)
        HL_convf = self.conv3(HL_conv)
        HH_convf = self.conv4(HH_conv)

        post_LL = LL + LL_convf
        post_LH = LH + LH_convf
        post_HL = HL + HL_convf
        post_HH = HH + HH_convf

        combined = tf.concat([post_LL, post_LH, post_HL, post_HH], 3)
        out = self.invwavelet(combined)

        return out

class TestNet2(tf.keras.Model):
    def __init__(self):
        super(TestNet2, self).__init__()
        self.my_initial = tf.initializers.he_uniform()
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        self.feature_num = 64

        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        
        #self.EAMa = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        #self.EAMb = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        #self.EAMc = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        #self.EAMd = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        
        self. conv10a = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10b = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        #self. conv10c = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        #self. conv10d = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)

        self.conv1 = layers.Conv2D(3, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv2 = layers.Conv2D(9, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)

    def call(self, input):
        wav= self.wavelet(input)
        LL = wav[:, :, :, 0:3]
        LLH = wav[:, :, :, 3:12]

        LL_conv = self.conv10a(wav)
        afterLL = self.conv1(LL_conv)

        LLH_conv = self.conv10b(wav)        
        afterLLH = self.conv2(LLH_conv)

        post_LL = LL + afterLL
        post_LLH = LLH + afterLLH


        combined = tf.concat([post_LL, post_LLH], 3)
        out = self.invwavelet(combined)

        return out


class TestNet3(tf.keras.Model):
    def __init__(self):
        super(TestNet3, self).__init__()
        self.my_initial = tf.initializers.he_uniform()
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        self.feature_num = 64

        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        
        #self.EAMa = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        #self.EAMb = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        #self.EAMc = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        #self.EAMd = EAMBlock(self.feature_num, self.my_initial, self.my_regular)
        
        self. conv10a = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10b = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10c = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10d = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)

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

        combined = tf.concat([aa, bb, cc, dd], 3)
        after = self.convAF(combined)
        out_ = self.invwavelet(after)
        out = input + out_
        return out