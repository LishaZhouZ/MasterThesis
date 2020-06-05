import numpy as np
import DnCNN_Feature_Attention as variants
import model_DnCNN as ds
from pathlib import Path

model = variants.RIDNet()#variants.DnCNN()
restore_ckptPath = Path("./tf_ckpts")
debug_mode = True
channel = 3
batch_size = variants.batch_size
patch_size = variants.patch_size
epochs = 50
record_step = 1
alpha = 0.0001
decay_lr = np.ones(epochs+1)
decay_lr[0:10]= alpha
decay_lr[10:20]= alpha/10
decay_lr[20:30] = alpha/100
decay_lr[30:40]= alpha/1000
decay_lr[40:50] = alpha/10000

#decay_lr[50:60] = alpha/50
#decay_lr[60:epochs+1] = alpha/100

checkpoint_directory = './tf_ckpts'

if debug_mode == True:
    batch_size = 16


