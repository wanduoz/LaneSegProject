
# linux
image_dir = '/root/data/LaneSeg/Image_Data/'
label_dir = '/root/data/LaneSeg/Gray_Label/'

train_list_dir = '/root/private/project_1/data/train.csv'
val_list_dir = '/root/private/project_1/data/val.csv'
test_list_dir = '/root/private/project_1/data/test.csv'

save_model_path = '/root/private/project_1/model'
pred_model_path = '/root/private/project_1/test_example'
ensemble_path = '/root/private/project_1/ensemble_result'

# window
# image_dir = 'G:\\LaneSegProject_final\\dataset\\Image_Data'
# label_dir = 'G:\\LaneSegProject_final\\dataset\\Gray_Label'
#
# train_list_dir = 'G:\\LaneSegProject\\dataset\\csv_file_all\\train.csv'
# val_list_dir = 'G:\\LaneSegProject\\dataset\\csv_file_all\\val.csv'
# test_list_dir = 'G:\\LaneSegProject\\dataset\\csv_file_all\\test.csv'
#
# save_model_path = 'G:\\LaneSegProject_final\\model'
# pred_model_path = 'G:\\LaneSegProject_final\\test_example'
# ensemble_path = 'G:\\LaneSegProject_final\\ensemble_result'

num_classes = 8
base_lr =  0.0005
weight_decay = 1.0e-4
crop_offset = 690

IMG_SIZE = (1024,384)#(1024, 384)
SUBMISSION_SIZE = [3384, 1710]

# 早停
patience = 10
epsilon = 1e-4

# train csv获取的每个类别的出现次数。用于计算权重（全数据集）
# cls_num = [5008701637, 63097603, 13996236, 5561706, 2457250, 51732355, 18755731, 5701450]
# cls_weight = [0.0002, 0.0172, 0.0775, 0.1949, 0.4412, 0.021, 0.0578, 0.1902]

# 去除road03数据集
cls_num = [542653221, 8033843, 1633291, 567013, 398999, 5209287, 1498464, 862970]
# 使用 inverse number normalization
cls_weight = [0.0003, 0.0177, 0.0871, 0.251, 0.3567, 0.0273, 0.095, 0.1649]