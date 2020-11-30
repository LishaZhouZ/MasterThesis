from model_utility import ConvBlock10_2, WaveletInvLayer, WaveletConvLayer
import tensorflow as tf
from tensorflow.keras import layers

class DualinBlock(layers.Layer):
    def __init__(self, feature_num, kernel_size, my_initial, my_regular):
        super(DualinBlock, self).__init__()
        
        self. conv10a = ConvBlock10_2(feature_num, kernel_size, my_initial, my_regular)
        self. conv10b = ConvBlock10_2(feature_num, kernel_size, my_initial, my_regular)
        self.conv1 = layers.Conv2D(12, kernel_size, padding = 'SAME',
                              kernel_initializer = my_initial, kernel_regularizer = my_regular)
        self.conv2 = layers.Conv2D(12, kernel_size, padding = 'SAME',
                              kernel_initializer = my_initial, kernel_regularizer = my_regular)
        self.scale1 = tf.Variable(0.1, trainable = True)
        self.scale2 = tf.Variable(0.1, trainable = True)
                              

    def call(self, input_main, input_aux):
        
        input_1st = tf.concat([input_main, input_aux],3)

        main_1 = self.conv10a(input_1st)
        main_2 = self.conv1(main_1)

        auxiliary_1 = self.conv10b(input_1st)
        auxiliary_2 = self.conv2(auxiliary_1)

        main_af = self.scale1*(input_main + main_2)
        auxiliary_af = self.scale2*(input_aux + auxiliary_2)



        return main_af, auxiliary_af


class DualinNet(tf.keras.Model):
    def __init__(self):
        super(DualinNet, self).__init__()
        self.my_initial = tf.random_normal_initializer(0, 0.01)
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)

        self.dualinBlock1 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock2 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock3 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock4 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.conv = layers.Conv2D(12, (3,3), padding = 'SAME',
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        
        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        

    def call(self, input):
        wav_in = self.wavelet(input)
        pixel_shifted = tf.nn.space_to_depth(input, 2, data_format='NHWC', name=None)
        
        main_1, aux_1 = self.dualinBlock1(wav_in, pixel_shifted)
        main_2, aux_2 = self.dualinBlock2(main_1, aux_1)
        main_3, aux_3 = self.dualinBlock3(main_2, aux_2)
        main_4, aux_4 = self.dualinBlock4(main_3, aux_3)
        
        combined = tf.concat([main_4, aux_4],3)

        after = self.conv(combined) + wav_in
        
        out = self.invwavelet(after)
        
        return out


