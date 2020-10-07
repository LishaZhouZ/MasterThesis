from model_utility import WaveletConvLayer, WaveletInvLayer, ConvBlock10, ConvConcatLayer
import tensorflow as tf
class HopeNet(tf.keras.Model):
  def __init__(self):
    super(HopeNet, self).__init__()
    self.my_initial = tf.initializers.he_normal()
    self.my_regular = tf.keras.regularizers.l2(l=0.0001)
    
    self. conv10a = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
    self. conv10b = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
    self. conv10c = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)
    self. conv10d = ConvBlock10(64, (3,3), self.my_initial, self.my_regular)

    self.before = ConvConcatLayer(120, (3,3), self.my_initial, self.my_regular)
    self.after = ConvConcatLayer(120, (3,3), self.my_initial, self.my_regular)
    #wavelet part
    self.wavelet = WaveletConvLayer()

    self.invwavelet = WaveletInvLayer()

  
  def call(self, inputs):

    wav = self.wavelet(inputs)  #
    # seperate to LL and HL,LH,HH part
    LL = wav[:, :, :, 0:3]
    LH = wav[:, :, :, 3:6]
    HL = wav[:, :, :, 6:9]
    HH = wav[:, :, :, 9:12]

    conLL = self.conv10a(LL)
    conLH = self.conv10b(LH)
    conHL = self.conv10c(HL)
    conHH = self.conv10d(HH)

    post_LL = LL + conLL
    post_LH = LH + conLH
    post_HL = HL + conHL
    post_HH = HH + conHH

    resultInter = tf.concat([post_LL, post_LH, post_HL, post_HH], 3)

    invwav = self.invwavelet(resultInter) 
    return invwav
    