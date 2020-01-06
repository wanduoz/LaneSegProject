

image_dir = '/root/data/LaneSeg/Image_Data/'
label_dir = '/root/data/LaneSeg/Gray_Label/'

train_list_dir = '/root/private/data/train.csv'
val_list_dir = '/root/private/data/val.csv'
test_list_dir = '/root/private/data/test.csv'

save_model_path = '/root/private/project_1/model'


num_classes = 8
log_iters = 100
base_lr =  6.0e-4
weight_decay = 1.0e-4
crop_offset = 690
save_model_iters = 2000
IMG_SIZE = [1536, 512]
SUBMISSION_SIZE = [3384, 1710]

use_pretrained = True

