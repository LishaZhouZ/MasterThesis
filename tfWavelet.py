import tensorflow as tf
import numpy as np

from PIL import Image
import time
import pywt
import matplotlib.pyplot as plt
from utils_py3_tfrecord import *
### def cdf9/7 wavlet taps
def cdf97():
    Lo_D = np.array([0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443,\
                            0.602949018236, 0.266864118443, -0.078223266529,-0.016864118443, \
                            0.026748757411])
    Hi_D = np.array([0.091271763114, -0.057543526229,-0.591271763114,1.11508705,\
                            -0.591271763114,-0.057543526229,0.091271763114,0,0])
    Lo_R = np.array([-0.091271763114,-0.057543526229,0.591271763114,1.11508705,\
                             0.591271763114 ,-0.057543526229,-0.091271763114,0,0])
    Hi_R = np.array([0.026748757411,0.016864118443,-0.078223266529,-0.266864118443,\
                             0.602949018236,-0.266864118443,-0.078223266529,0.016864118443,\
                             0.026748757411])
    cdf97 = {
            'Lo_D':Lo_D,
            'Hi_D':Hi_D,
            'Lo_R':Lo_R,
            'Hi_R':Hi_R
            }
    return cdf97


### Create an DWT object 2d discrete wavelet transform
class tfWavelet(object):
    """
    class that provides a 2D DWT implimentation in TensorFlow.
    uses the Mallat Algorithm for 2d decomposition
    """

    def __init__(self, wavelet=cdf97()):
        # calculate the pad length from filter length
        self.tap_length = len(wavelet['Lo_D'])
        self.pad_length = self.tap_length//2*2

        ### parse wavelet taps
        # low pass tap for decomposition
        self.Lo_D = tf.convert_to_tensor(wavelet['Lo_D'],dtype=tf.float32)
        # High pass tap for decomposition
        self.Hi_D = tf.convert_to_tensor(wavelet['Hi_D'],dtype=tf.float32)
        # low pass tap for reconstruction
        self.Lo_R = tf.convert_to_tensor(wavelet['Lo_R'],dtype=tf.float32)
        # High pass tap for reconstruction
        self.Hi_R = tf.convert_to_tensor(wavelet['Hi_R'],dtype=tf.float32)

        # capture coeff shape for multi_DWT
        self.coeff_shapes = []


    def multi_DWT(self, data):
        """
        multi level 2D DWT decomposition

        INPUT DATA FORMAT:
        [batch x dim1 x dim2 x channels]

        RETURNS:
        approx: [batch, dim1, dim2, channels]
        detail: [batch, dim1, dim2, [cH(channels 1 to n),cV,cD], level]

        NOTE:
        to make the detail tensor constant size, the higher level
        detail coeffs are resized ()to size of the first level
        """

        # track shape of coeffs
        batch_num, dim1_num, dim2_num, channel_num = data.get_shape().as_list()
        self.coeff_shapes = []

        # first DWT decomposition
        approx, detail = self.DWT(data)
        detail = tf.concat(tf.split(detail,3,axis=-1), axis=-2)

        # track shape
        _, out_dim1, out_dim2, _, _ = detail.get_shape().as_list()
        self.coeff_shapes.insert(0,[out_dim1,out_dim2])

        # further recursive DWT decomposition
        while min(dim1_num, dim2_num) >= (2*self.tap_length-self.pad_length)*2 :
            approx, detail_ = self.DWT(approx)
            detail_ = tf.squeeze(tf.concat(tf.split(detail_,3,axis=-1), axis=-2))

            # track shape
            _, dim1_num, dim2_num, _ = approx.get_shape().as_list()
            self.coeff_shapes.insert(0,[dim1_num,dim2_num])

            # upscale
            detail_ = tf.image.resize_bicubic(detail_, [out_dim1,out_dim2], align_corners=True)

            detail_ = tf.expand_dims(detail_, axis=-1)
            detail = tf.concat([detail_, detail], axis=-1)

        return approx, detail


    def multi_iDWT(self, approx, detail):
        """
        multi level 2D DWT reconstruction

        INPUT DATA FORMAT:
        approx: [batch, dim1, dim2, channels]
        detail: [batch, dim1, dim2, [cH(channels 1 to n),cV,cD], level]

        RETURNS:
        [batch x dim1 x dim2 x channels]
        """

        batch_num, dim1_num, dim2_num, channel_num, levels = detail.get_shape().as_list()

        # recursive iDWT reconstruction
        for i,coeff_shape in enumerate(self.coeff_shapes):
            # get detail level
            detail_ = detail[:,:,:,:,i]

            # downscale
            detail_ = tf.image.resize_bicubic(detail_, coeff_shape, align_corners=True)

            detail_ = tf.stack(tf.split(detail_,3,axis=-1),axis=-1)
            approx = self.iDWT(approx, detail_)

        return approx


    def DWT(self, data):
        """
        single level 2D DWT decomposition

        INPUT DATA FORMAT:
        [batch x dim1 x dim2 x channels]

        RETURNS:
        approx: [batch, dim1, dim2, channels]
        detail: [batch, dim1, dim2, channels, [cH,cV,cD]]
        """

        # pad data
        paddings = [[0,0],[self.pad_length,self.pad_length], \
                        [self.pad_length,self.pad_length],[0,0]]
        data = tf.pad(data, paddings, 'SYMMETRIC')

        # horizontal decomposition
        data_Lo = self.conv_axis_and_downsample(data, 1, self.Lo_D)
        data_Hi = self.conv_axis_and_downsample(data, 1, self.Hi_D)

        # vertical decomposition
        cA = self.conv_axis_and_downsample(data_Lo, 0, self.Lo_D)
        cH = self.conv_axis_and_downsample(data_Lo, 0, self.Hi_D)
        cV = self.conv_axis_and_downsample(data_Hi, 0, self.Lo_D)
        cD = self.conv_axis_and_downsample(data_Hi, 0, self.Hi_D)

        # pack coeffs
        approx = cA
        detail = tf.stack([cH,cV,cD], axis=-1)
        return approx, detail


    def iDWT(self, approx, detail):
        """
        single level 2D inverse DWT reconstruction

        INPUT DATA FORMAT:
        approx: [batch, dim1, dim2, channels]
        detail: [batch, dim1, dim2, channels, [cH,cV,cD]]

        RETURNS:
        [batch x dim1 x dim2 x channels]
        """
        # check shapes match
        _,dim1,dim2,_,_ = detail.get_shape().as_list()
        _,dim1_a,dim2_a,_ = approx.get_shape().as_list()
        if dim1 != dim1_a:
            approx = approx[:,1:1+dim1,:,:]
        if dim2 != dim2_a:
            approx = approx[:,:,1:1+dim2,:]

        # unpack coeffs
        cA = approx
        cH,cV,cD = tf.unstack(detail, axis=-1)

        # vertical decomposition
        data_Lo = self.upsample_and_conv_axis(cA, 0, self.Lo_R) + \
                        self.upsample_and_conv_axis(cH, 0, self.Hi_R)
        data_Hi = self.upsample_and_conv_axis(cV, 0, self.Lo_R) + \
                        self.upsample_and_conv_axis(cD, 0, self.Hi_R)

        # horizontal decomposition
        data = self.upsample_and_conv_axis(data_Lo, 1, self.Lo_R) + \
                    self.upsample_and_conv_axis(data_Hi, 1, self.Hi_R)

        # unpad data
        data = data[:,self.pad_length:-self.pad_length,self.pad_length:-self.pad_length,:]
        return data


    def conv_axis_and_downsample(self, data, axis, filter):
        """
        1D convolution followed by decimation along one axis
        implemented as a 2D separable convolution and matmul

        INPUT DATA FORMAT:
        [batch x dim1 x dim2 x channels]
        """

        assert axis==0 or axis==1, "axis must be 0 or 1"
        batch_num, dim1_num, dim2_num, channel_num = data.get_shape().as_list()

        if axis == 0:
            # pad dim1
            # data_pad = tf.pad(data, [[0,0],[self.pad_length,self.pad_length],[0,0],[0,0]], 'REFLECT')
            # make filter 2d
            filter = tf.expand_dims(filter,axis=1)

        else:
            # pad dim 2
            # data_pad = tf.pad(data, [[0,0],[0,0],[self.pad_length,self.pad_length],[0,0]], 'REFLECT')
            # make filter 2d
            filter = tf.expand_dims(filter,axis=0)

        ### make 4D depthwise for separable conv
        d_filt = tf.stack([filter for i in range(channel_num)], axis=-1)
        d_filt = tf.expand_dims(d_filt,axis=-1)

        ### convolution
        data_conv = tf.nn.depthwise_conv2d(data,d_filt,[1,1,1,1],'VALID')

        ### downsample via matrix multiplication
        if axis == 0:
            data_downsampled = data_conv[:,5:-4:2,:,:]
            """
            size = dim1_num
            downsample_mat = tf.eye(size)
            downsample_mat = tf.strided_slice(downsample_mat,[0,0],[size,size],[1,2])
            downsample_mat = tf.reverse(downsample_mat,[0])
            downsample_mat = tf.reverse(downsample_mat,[1])

            def matmul0(input):
                input_T = tf.transpose(input) # channels first
                output_T = tf.map_fn(lambda i: tf.linalg.matmul(i, downsample_mat), input_T, parallel_iterations=40)
                return tf.transpose(output_T)
            data_downsampled = tf.map_fn(lambda i: matmul0(i), data_conv, parallel_iterations=40)
            """
        else:
            data_downsampled = data_conv[:,:,5:-4:2,:]
            """
            size = dim2_num
            downsample_mat = tf.eye(size)
            downsample_mat = tf.strided_slice(downsample_mat,[0,0],[size,size],[2,1])
            downsample_mat = tf.reverse(downsample_mat,[0])
            downsample_mat = tf.reverse(downsample_mat,[1])

            def matmul1(input):
                input_T = tf.transpose(input) # channels first
                output_T = tf.map_fn(lambda i: tf.linalg.matmul(downsample_mat, i), input_T, parallel_iterations=40)
                return tf.transpose(output_T)
            data_downsampled = tf.map_fn(lambda i: matmul1(i), data_conv, parallel_iterations=40)
        """
        return data_downsampled


    def upsample_and_conv_axis(self, data, axis, filter):
        """
        1D convolution followed by expansion by zero striping along one axis

        INPUT DATA FORMAT:
        [batch x dim1 x dim2 x channels]
        """

        batch_num, dim1_num, dim2_num, channel_num = data.get_shape().as_list()
        assert axis==0 or axis==1, "axis must be 0 or 1"
        data_zero = tf.zeros_like(data)
        data_temp = tf.concat([data,data_zero,data_zero,data_zero], 3)
        data_temp_up = tf.depth_to_space(data_temp,2)
        ### upsample via matrix multiplication
        if axis == 0:
            data_upsampled = data_temp_up[:,:,0::2,:]
            """
            size = dim1_num*2
            upsample_mat = tf.eye(size)
            upsample_mat = tf.strided_slice(upsample_mat,[0,0],[size,size],[2,1])

            def matmul0(input):
                input_T = tf.transpose(input) # channels first
                output_T = tf.map_fn(lambda i: tf.linalg.matmul(i, upsample_mat), input_T, parallel_iterations=40)
                return tf.transpose(output_T)
            data_upsampled = tf.map_fn(lambda i: matmul0(i), data, parallel_iterations=40)
            """
        else:
            data_upsampled = data_temp_up[:,0::2,:,:]
            """
            size = dim2_num*2
            upsample_mat = tf.eye(size)
            upsample_mat = tf.strided_slice(upsample_mat,[0,0],[size,size],[1,2])

            def matmul1(input):
                input_T = tf.transpose(input) # channels first
                output_T = tf.map_fn(lambda i: tf.linalg.matmul(upsample_mat, i), input_T, parallel_iterations=40)
                return tf.transpose(output_T)
            data_upsampled = tf.map_fn(lambda i: matmul1(i), data, parallel_iterations=40)
            """
        ### pad and make filters 2D
        if axis == 0:
            # pad data
            # data_pad = tf.pad(data_upsampled, [[0,0],[self.pad_length,self.pad_length],[0,0],[0,0]], 'REFLECT')
            # make filter 2d
            filter = tf.expand_dims(filter,axis=1)
        else:
            # pad data
            # data_pad = tf.pad(data_upsampled, [[0,0],[0,0],[self.pad_length,self.pad_length],[0,0]], 'REFLECT')
            # make filter 2d
            filter = tf.expand_dims(filter,axis=0)

        ### make 4D depthwise filters for separable conv
        d_filt = tf.stack([filter for i in range(channel_num)], axis=-1)
        d_filt = tf.expand_dims(d_filt,axis=-1)

        ### convolution
        data_conv = tf.nn.depthwise_conv2d(data_upsampled,d_filt,[1,1,1,1],'VALID')
        return data_conv


    def My_DWT97(self, data):
        """
        single level 2D DWT decomposition

        INPUT DATA FORMAT:
        [batch x dim1 x dim2 x channels]

        RETURNS:
        coeffs: [batch x dim1//2 x dim2//2 x channels*4]
        """

        # pad data
        paddings = [[0,0],[self.pad_length,self.pad_length],[self.pad_length,self.pad_length],[0,0]]
        #data = tf.pad(data, paddings, 'SYMMETRIC')
        data = self.pad_circular(data, paddings)
        # horizontal decomposition
        data_Lo = self.conv_axis_and_downsample(data, 1, self.Lo_D)
        data_Hi = self.conv_axis_and_downsample(data, 1, self.Hi_D)
        # vertical decomposition
        cA = self.conv_axis_and_downsample(data_Lo, 0, self.Lo_D)
        cH = self.conv_axis_and_downsample(data_Lo, 0, self.Hi_D)
        cV = self.conv_axis_and_downsample(data_Hi, 0, self.Lo_D)
        cD = self.conv_axis_and_downsample(data_Hi, 0, self.Hi_D)
        # pack coeffs
        coeffs = tf.concat([cA,cH,cV,cD], axis=3)
        return coeffs#[:,self.pad_length//2:-self.pad_length//2,self.pad_length//2:-self.pad_length//2,:]

    def My_iDWT97(self, coeffs):
        """
        single level 2D inverse DWT reconstruction

        INPUT DATA FORMAT:
        coeffs: [batch x dim1//2 x dim2//2 x channels*4]

        RETURNS:
        [batch x dim1 x dim2 x channels]
        """
        paddings = [[0,0],[self.pad_length,self.pad_length],[self.pad_length,self.pad_length],[0,0]]
        #coeffs = tf.pad(coeffs, paddings, 'SYMMETRIC')
        coeffs = self.pad_circular(coeffs, paddings)
        # unpack coeffs
        out_channel = coeffs.shape[3]//4
        cA = coeffs[:, :, :, 0:out_channel]
        cH = coeffs[:, :, :, out_channel:out_channel * 2]
        cV = coeffs[:, :, :, out_channel * 2:out_channel * 3]
        cD = coeffs[:, :, :, out_channel * 3:out_channel * 4]

        # vertical decomposition
        data_Lo = self.upsample_and_conv_axis(cA, 0, self.Lo_R) + self.upsample_and_conv_axis(cH, 0, self.Hi_R)
        data_Hi = self.upsample_and_conv_axis(cV, 0, self.Lo_R) + self.upsample_and_conv_axis(cD, 0, self.Hi_R)

        # horizontal decomposition
        data = self.upsample_and_conv_axis(data_Lo, 1, self.Lo_R) + self.upsample_and_conv_axis(data_Hi, 1, self.Hi_R)

        # unpad data
        data = data[:,12:-12,12:-12,:]
        return data
    
    def pad_circular(self, image, paddings=[[0,0],[8,8],[8,8],[0,0]]):
        #paddings = [[0,0],[8,8],[8,8],[0,0]]
        # batch_num, m, n, channel_num = image.get_shape().as_list()
        batch_num = tf.shape(image)[0]
        m = tf.shape(image)[1]
        n = tf.shape(image)[2]
        channel_num = tf.shape(image)[3]
        pad_left  = image[:,:,n-paddings[2][0]:n,:]
        pad_right = image[:,:,0:paddings[2][1],:]
        # pad_left  = image[:,:,n-8:n,:]
        # pad_right = image[:,:,0:8,:]
        image  = tf.concat([pad_left,image,pad_right], 2)
        pad_up = image[:,m-paddings[1][0]:m,:,:]
        pad_down = image[:,0:paddings[1][1],:,:]
        # pad_up = image[:,m-8:m,:,:]
        # pad_down = image[:,0:8,:,:]
        return tf.concat([pad_up,image,pad_down], 1)

#Just for debug
if __name__ == '__main__':
    # load images
    a = Image.open('lena.bmp')
    b = Image.open('Pepper.bmp')
    c = Image.open('satpic1.bmp')
    X_a = np.array(a).astype('float32')
    X_b = np.array(b).astype('float32')
    X_c = np.array(c).astype('float32')
    X_t = np.zeros((3, 512, 512, 3), dtype='float32')
    X_t[0, :, :, 0:3] = X_a[:, :, :]
    X_t[1, :, :, 0:3] = X_b[:, :, :]
    X_t[2, :, :, 0:3] = X_c[:, :, :]
    my_dwt = tfWavelet()
    X_tf = tf.convert_to_tensor(X_t)
    # convert to tensor
    sess = tf.Session()
    inputs = tf.placeholder(tf.float32, [None, None, None, 3], name='inputs')
    dwt97coefs = my_dwt.My_DWT97(inputs)
    print(dwt97coefs.shape)
    outputs_mex = my_dwt.My_iDWT97(dwt97coefs)
    print(outputs_mex.shape)
    sess.run(tf.global_variables_initializer())
    time_start = time.time()
    dwt97coefs = sess.run(dwt97coefs, feed_dict={inputs: X_t})
    outputs_mex = sess.run(outputs_mex, feed_dict={inputs: X_t})
    outputs_mex = np.round(np.clip(outputs_mex, 0,255))
    time_end = time.time()
    vis = 0
    print('totally cost', time_end - time_start)
    print('PSNR', imcpsnr(X_t[vis, :, :, 0:3],outputs_mex[vis, :, :, 0:3],255))
    # show the decomposition images
    plt.figure()
    plt.imshow(X_t[vis, :, :, 0:3].astype('uint8'), cmap='gray')
    plt.figure()
    plt.imshow(outputs_mex[vis, :, :, 0:3].astype('uint8'), cmap='gray')
    plt.show()