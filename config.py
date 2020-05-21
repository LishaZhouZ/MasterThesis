import numpy as np
import model_DnCNN

debug_mode = False
channel = 3
batch_size = model_DnCNN.batch_size
patch_size = model_DnCNN.patch_size
epochs = 50
record_step = 1
alpha = 10
decay_lr = np.ones(epochs+1)
decay_lr[0:5]= alpha
decay_lr[5:10]= alpha/2
decay_lr[10:15] = alpha/10
decay_lr[15:20]= alpha/50
decay_lr[20:25] = alpha/100
decay_lr[25:30]= alpha/500
decay_lr[30:40] = alpha/1000
decay_lr[40:50] = alpha/10000

#decay_lr[50:60] = alpha/50
#decay_lr[60:epochs+1] = alpha/100

checkpoint_directory = './tf_ckpts'

if debug_mode == True:
    batch_size = 8


