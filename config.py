

image_dir = '/root/data/LaneSeg/Image_Data/'
label_dir = '/root/data/LaneSeg/Gray_Label/'

train_list_dir = '/root/private/project_1/data/train.csv'
val_list_dir = '/root/private/project_1/data/val.csv'
test_list_dir = '/root/private/project_1/data/test.csv'

save_model_path = '/root/private/project_1/model'


num_classes = 8
base_lr =  6.0e-4
weight_decay = 1.0e-4
crop_offset = 690

IMG_SIZE = (256,256)#(1024, 384)
SUBMISSION_SIZE = [3384, 1710]

