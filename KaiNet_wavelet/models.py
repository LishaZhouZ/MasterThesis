from model_utility import ConvBlock10, WaveletInvLayer, WaveletConvLayer, ConvBlock
from DnCNN_Feature_Attention import EAMBlock
import tensorflow as tf
from tensorflow.keras import layers

#best with pz 128, 27.29
class KaiNet(tf.keras.Model):
    def __init__(self):
        super(KaiNet, self).__init__()
        self.my_initial = tf.initializers.he_normal()
        
        self.my_regular = tf.keras.regularizers.l2(l=0.0005)
        self.feature_num = 64

        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        self. conv10a = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10b = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10c = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10d = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)

        self.conv1 = layers.Conv2D(12, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv2 = layers.Conv2D(3, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv3 = layers.Conv2D(9, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)
        self.conv4 = layers.Conv2D(12, (3,3), padding = 'SAME', kernel_initializer=self.my_initial, kernel_regularizer=self.my_regular)

    def call(self, input):
        wav= self.wavelet(input)
        #first stage
        wav_conv = self.conv10a(wav)
        after_wav = self.conv1(wav_conv )+ wav

        LL = after_wav[:, :, :, 0:3]
        LHH = after_wav[:, :, :, 3:12]
        
        #second stage
        LL_conv = self.conv10b(LL)
        LL_convf = self.conv2(LL_conv) + LL

        LHH_conv = self.conv10c(LHH)
        LHH_convf = self.conv3(LHH_conv) + LHH

        #third stage 
        
        combined = tf.concat([LL_convf, LHH_convf], 3)
        combined_conv = self.conv10d(combined)
        combined_convf = self.conv4(combined_conv) + combined

        out = self.invwavelet(combined_convf)

        return out

