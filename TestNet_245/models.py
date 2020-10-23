from model_utility import ConvBlock10, WaveletInvLayer, WaveletConvLayer, ConvBlock
import tensorflow as tf
from tensorflow.keras import layers

#best with pz 128, 27.29
class TestNet(tf.keras.Model):
    def __init__(self):
        super(TestNet, self).__init__()
        self.my_initial = tf.initializers.Constant(0.1)
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        self.feature_num = 64

        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        self. conv10a = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10b = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10c = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
        self. conv10d = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)


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
        wav = self.wavelet(input)
        wav_arranged = tf.nn.space_to_depth(wav, 2, data_format='NHWC', name=None)
        
        
        LL = wav[:, :, :, 0:3] #64,64
        #LLH = wav[:, :, :, 3:12]

        wav2 = self.wavelet(LL)
        LL2 = wav2[:, :, :, 0:3] #32
        LLH2 = wav2[:, :, :, 3:12]
        
        LL2_conv = self.conv10a(wav_arranged)
        LL2_convf = self.conv1(LL2_conv)

        LLH2_conv = self.conv10b(wav_arranged)
        LLH2_convf= self.conv2(LLH2_conv)

        post_LL = LL2 + LL2_convf
        post_LLH = LLH2 + LLH2_convf
        
        combined = tf.concat([post_LL, post_LLH], 3)
        out = self.invwavelet(combined)

        return out