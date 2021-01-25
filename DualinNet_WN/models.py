from model_utility import ConvBlock10_2, WaveletInvLayer, WaveletConvLayer
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

class DualinBlock(layers.Layer):
    def __init__(self, feature_num, kernel_size, my_initial, my_regular):
        super(DualinBlock, self).__init__()
        self.conv1_main = layers.Conv2D(feature_num, kernel_size, padding = 'SAME',
            kernel_initializer=my_initial,kernel_regularizer=my_regular)# 
        self.bn_main = tfa.layers.WeightNormalization()
        self.relu_main = layers.ReLU()
        self.conv2_main = layers.Conv2D(feature_num, kernel_size, padding = 'SAME',
            kernel_initializer=my_initial,kernel_regularizer=my_regular)# 
        
        self.conv1_aux = layers.Conv2D(feature_num, kernel_size, padding = 'SAME',
            kernel_initializer=my_initial,kernel_regularizer=my_regular)# 
        self.bn_aux = tfa.layers.WeightNormalization()
        self.relu_aux = layers.ReLU()
        self.conv2_aux = layers.Conv2D(feature_num, kernel_size, padding = 'SAME',
            kernel_initializer=my_initial,kernel_regularizer=my_regular)# 
        
        self.scale1 = tf.Variable(0.1, trainable = True)
        self.scale2 = tf.Variable(0.1, trainable = True)
                              

    def call(self, input_main, input_aux):
        
        input_1st = tf.concat([input_main, input_aux],3)

        main_1 = self.conv1_main(input_1st)
        main_2 = self.bn_main(main_1)
        main_3 = self.relu_main(main_2)
        main_4 = self.conv2_main(main_3)


        aux_1 = self.conv1_aux(input_1st)
        aux_2 = self.bn_aux(aux_1)
        aux_3 = self.relu_aux(aux_2)
        aux_4 = self.conv2_aux(aux_3)

        main_af = self.scale1*main_4 + input_main
        auxiliary_af = self.scale2*aux_4 + input_aux

        return main_af, auxiliary_af


class DualinNet(tf.keras.Model):
    def __init__(self):
        super(DualinNet, self).__init__()
        self.my_initial = tf.random_normal_initializer(0, 0.01)
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        
        self.conv1 = layers.Conv2D(64, (3,3), padding = 'SAME',  
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        self.conv2 = layers.Conv2D(64, (3,3), padding = 'SAME',
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        self.dualinBlock1 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock2 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock3 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock4 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock5 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock6 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock7 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock8 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock9 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock10 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)

        self.conv = layers.Conv2D(12, (3,3), padding = 'SAME',
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        
        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        

    def call(self, input):
        
        wav_in = self.wavelet(input)
        wav_input = self.conv1(wav_in)

        pixel_shifted = tf.nn.space_to_depth(input, 2, data_format='NHWC', name=None)
        pixel_input = self.conv2(pixel_shifted)

        main_1, aux_1 = self.dualinBlock1(wav_input, pixel_input)
        main_2, aux_2 = self.dualinBlock2(main_1, aux_1)
        main_3, aux_3 = self.dualinBlock3(main_2, aux_2)
        main_4, aux_4 = self.dualinBlock4(main_3, aux_3)
        main_5, aux_5 = self.dualinBlock5(main_4, aux_4)
        main_6, aux_6 = self.dualinBlock6(main_5, aux_5)
        main_7, aux_7 = self.dualinBlock7(main_6, aux_6)
        main_8, aux_8 = self.dualinBlock8(main_7, aux_7)
        main_9, aux_9 = self.dualinBlock9(main_8, aux_8)
        main_10, aux_10 = self.dualinBlock10(main_9, aux_9)

        combined = tf.concat([main_10, aux_10],3)

        after = self.conv(combined) + wav_in
        
        out = self.invwavelet(after)
        
        return out

class DualinNet_fm64_20(tf.keras.Model):
    def __init__(self):
        super(DualinNet_fm64_20, self).__init__()
        self.my_initial = tf.random_normal_initializer(0, 0.01)
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        
        self.conv1 = layers.Conv2D(64, (3,3), padding = 'SAME',  
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        self.conv2 = layers.Conv2D(64, (3,3), padding = 'SAME',
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        self.dualinBlock1 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock2 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock3 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock4 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock5 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock6 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock7 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock8 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock9 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock10 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock11 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock12 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock13 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock14 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock15 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock16 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock17 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock18 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock19 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock20 = DualinBlock(64, (3,3), self.my_initial, self.my_regular)
        self.conv = layers.Conv2D(12, (3,3), padding = 'SAME',kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        
        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        

    def call(self, input):
        
        wav_in = self.wavelet(input)
        wav_input = self.conv1(wav_in)

        pixel_shifted = tf.nn.space_to_depth(input, 2, data_format='NHWC', name=None)
        pixel_input = self.conv2(pixel_shifted)

        main_1, aux_1 = self.dualinBlock1(wav_input, pixel_input)
        main_2, aux_2 = self.dualinBlock2(main_1, aux_1)
        main_3, aux_3 = self.dualinBlock3(main_2, aux_2)
        main_4, aux_4 = self.dualinBlock4(main_3, aux_3)
        main_5, aux_5 = self.dualinBlock5(main_4, aux_4)
        main_6, aux_6 = self.dualinBlock6(main_5, aux_5)
        main_7, aux_7 = self.dualinBlock7(main_6, aux_6)
        main_8, aux_8 = self.dualinBlock8(main_7, aux_7)
        main_9, aux_9 = self.dualinBlock9(main_8, aux_8)
        main_10, aux_10 = self.dualinBlock10(main_9, aux_9)
        main_11, aux_11 = self.dualinBlock11(main_10, aux_10)
        main_12, aux_12 = self.dualinBlock12(main_11, aux_11)
        main_13, aux_13 = self.dualinBlock13(main_12, aux_12)
        main_14, aux_14 = self.dualinBlock14(main_13, aux_13)
        main_15, aux_15 = self.dualinBlock15(main_14, aux_14)
        main_16, aux_16 = self.dualinBlock16(main_15, aux_15)
        main_17, aux_17 = self.dualinBlock17(main_16, aux_16)
        main_18, aux_18 = self.dualinBlock18(main_17, aux_17)
        main_19, aux_19 = self.dualinBlock19(main_18, aux_18)
        main_20, aux_20 = self.dualinBlock20(main_19, aux_19)

        combined = tf.concat([main_20, aux_20],3)

        after = self.conv(combined)
        
        out = self.invwavelet(after) + input
        
        return out



class DualinNet_96(tf.keras.Model):
    def __init__(self):
        super(DualinNet_96, self).__init__()
        self.my_initial = tf.random_normal_initializer(0, 0.01)
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        
        self.conv1 = layers.Conv2D(96, (3,3), padding = 'SAME',  
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        self.conv2 = layers.Conv2D(96, (3,3), padding = 'SAME',
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        self.dualinBlock1 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock2 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock3 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock4 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock5 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock6 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock7 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock8 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock9 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock10 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.conv = layers.Conv2D(12, (3,3), padding = 'SAME',kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        
        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        

    def call(self, input):
        
        wav_in = self.wavelet(input)
        wav_input = self.conv1(wav_in)

        pixel_shifted = tf.nn.space_to_depth(input, 2, data_format='NHWC', name=None)
        pixel_input = self.conv2(pixel_shifted)

        main_1, aux_1 = self.dualinBlock1(wav_input, pixel_input)
        main_2, aux_2 = self.dualinBlock2(main_1, aux_1)
        main_3, aux_3 = self.dualinBlock3(main_2, aux_2)
        main_4, aux_4 = self.dualinBlock4(main_3, aux_3)
        main_5, aux_5 = self.dualinBlock5(main_4, aux_4)
        main_6, aux_6 = self.dualinBlock6(main_5, aux_5)
        main_7, aux_7 = self.dualinBlock7(main_6, aux_6)
        main_8, aux_8 = self.dualinBlock8(main_7, aux_7)
        main_9, aux_9 = self.dualinBlock9(main_8, aux_8)
        main_10, aux_10 = self.dualinBlock10(main_9, aux_9)

        combined = tf.concat([main_10, aux_10],3)

        after = self.conv(combined) + wav_in
        
        out = self.invwavelet(after)
        
        return out


class DualinNet_fm96_20(tf.keras.Model):
    def __init__(self):
        super(DualinNet_fm96_20, self).__init__()
        self.my_initial = tf.random_normal_initializer(0, 0.01)
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        
        self.conv1 = layers.Conv2D(96, (3,3), padding = 'SAME',  
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        self.conv2 = layers.Conv2D(96, (3,3), padding = 'SAME',
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        self.dualinBlock1 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock2 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock3 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock4 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock5 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock6 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock7 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock8 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock9 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock10 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock11 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock12 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock13 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock14 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock15 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock16 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock17 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock18 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock19 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock20 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.conv = layers.Conv2D(12, (3,3), padding = 'SAME',kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        
        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        

    def call(self, input):
        
        wav_in = self.wavelet(input)
        wav_input = self.conv1(wav_in)

        pixel_shifted = tf.nn.space_to_depth(input, 2, data_format='NHWC', name=None)
        pixel_input = self.conv2(pixel_shifted)

        main_1, aux_1 = self.dualinBlock1(wav_input, pixel_input)
        main_2, aux_2 = self.dualinBlock2(main_1, aux_1)
        main_3, aux_3 = self.dualinBlock3(main_2, aux_2)
        main_4, aux_4 = self.dualinBlock4(main_3, aux_3)
        main_5, aux_5 = self.dualinBlock5(main_4, aux_4)
        main_6, aux_6 = self.dualinBlock6(main_5, aux_5)
        main_7, aux_7 = self.dualinBlock7(main_6, aux_6)
        main_8, aux_8 = self.dualinBlock8(main_7, aux_7)
        main_9, aux_9 = self.dualinBlock9(main_8, aux_8)
        main_10, aux_10 = self.dualinBlock10(main_9, aux_9)
        main_11, aux_11 = self.dualinBlock11(main_10, aux_10)
        main_12, aux_12 = self.dualinBlock12(main_11, aux_11)
        main_13, aux_13 = self.dualinBlock13(main_12, aux_12)
        main_14, aux_14 = self.dualinBlock14(main_13, aux_13)
        main_15, aux_15 = self.dualinBlock15(main_14, aux_14)
        main_16, aux_16 = self.dualinBlock16(main_15, aux_15)
        main_17, aux_17 = self.dualinBlock17(main_16, aux_16)
        main_18, aux_18 = self.dualinBlock18(main_17, aux_17)
        main_19, aux_19 = self.dualinBlock19(main_18, aux_18)
        main_20, aux_20 = self.dualinBlock20(main_19, aux_19)

        combined = tf.concat([main_20, aux_20],3)

        after = self.conv(combined) + wav_in
        
        out = self.invwavelet(after)
        
        return out


class DualinNet_fm96_30(tf.keras.Model):
    def __init__(self):
        super(DualinNet_fm96_30, self).__init__()
        self.my_initial = tf.random_normal_initializer(0, 0.01)
        
        self.my_regular = tf.keras.regularizers.l2(l=0.000001)
        
        self.conv1 = layers.Conv2D(96, (3,3), padding = 'SAME',  
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        self.conv2 = layers.Conv2D(96, (3,3), padding = 'SAME',
                              kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        self.dualinBlock1 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock2 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock3 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock4 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock5 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock6 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock7 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock8 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock9 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock10 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock11 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock12 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock13 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock14 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock15 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock16 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock17 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock18 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock19 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock20 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock21 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock22 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock23 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock24 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.dualinBlock25 = DualinBlock(96, (3,3), self.my_initial, self.my_regular)
        self.conv = layers.Conv2D(12, (3,3), padding = 'SAME',kernel_initializer = self.my_initial, kernel_regularizer = self.my_regular)
        
        self.wavelet = WaveletConvLayer()
        self.invwavelet = WaveletInvLayer()
        

    def call(self, input):
        
        wav_in = self.wavelet(input)
        wav_input = self.conv1(wav_in)

        pixel_shifted = tf.nn.space_to_depth(input, 2, data_format='NHWC', name=None)
        pixel_input = self.conv2(pixel_shifted)

        main_1, aux_1 = self.dualinBlock1(wav_input, pixel_input)
        main_2, aux_2 = self.dualinBlock2(main_1, aux_1)
        main_3, aux_3 = self.dualinBlock3(main_2, aux_2)
        main_4, aux_4 = self.dualinBlock4(main_3, aux_3)
        main_5, aux_5 = self.dualinBlock5(main_4, aux_4)
        main_6, aux_6 = self.dualinBlock6(main_5, aux_5)
        main_7, aux_7 = self.dualinBlock7(main_6, aux_6)
        main_8, aux_8 = self.dualinBlock8(main_7, aux_7)
        main_9, aux_9 = self.dualinBlock9(main_8, aux_8)
        main_10, aux_10 = self.dualinBlock10(main_9, aux_9)
        main_11, aux_11 = self.dualinBlock11(main_10, aux_10)
        main_12, aux_12 = self.dualinBlock12(main_11, aux_11)
        main_13, aux_13 = self.dualinBlock13(main_12, aux_12)
        main_14, aux_14 = self.dualinBlock14(main_13, aux_13)
        main_15, aux_15 = self.dualinBlock15(main_14, aux_14)
        main_16, aux_16 = self.dualinBlock16(main_15, aux_15)
        main_17, aux_17 = self.dualinBlock17(main_16, aux_16)
        main_18, aux_18 = self.dualinBlock18(main_17, aux_17)
        main_19, aux_19 = self.dualinBlock19(main_18, aux_18)
        main_20, aux_20 = self.dualinBlock20(main_19, aux_19)
        main_21, aux_21 = self.dualinBlock21(main_20, aux_20)
        main_22, aux_22 = self.dualinBlock22(main_21, aux_21)
        main_23, aux_23 = self.dualinBlock23(main_22, aux_22)
        main_24, aux_24 = self.dualinBlock24(main_23, aux_23)
        main_25, aux_25 = self.dualinBlock25(main_24, aux_24)


        combined = tf.concat([main_25, aux_25],3)

        after = self.conv(combined) + wav_in
        
        out = self.invwavelet(after)
        
        return out