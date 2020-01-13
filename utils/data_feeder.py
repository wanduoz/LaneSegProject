import cv2
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.label_process import encode_labels
from utils.image_process import crop_resize_data
from utils.image_process import ImageAug, DeformAug, ScaleAug, CutOut, ToTensor

# 创建类。导入label预处理和image预处理函数。返回torch Dataset类
class LaneDataset(Dataset):
    def __init__(self, csv_file, image_size, crop_offset, transform):
        super(LaneDataset, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.image = self.data["image"].values
        self.label = self.data["label"].values
        self.image_size = image_size
        self.crop_offset = crop_offset
        self.transform = transform

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):

        ori_image = cv2.imread(self.image[idx]) # 3维HWC
        ori_mask = cv2.imread(self.label[idx], cv2.IMREAD_GRAYSCALE) # 2维HW
        train_img, train_mask = crop_resize_data(ori_image, self.image_size, self.crop_offset, label=ori_mask)
        # Encode
        train_mask = encode_labels(train_mask)
        sample = [train_img.copy(), train_mask.copy()]
        if self.transform:
            sample = self.transform(sample)
        return sample

# 返回batchdata的generator
def batch_data_generator(csv_file, is_train, batch_size, image_size, crop_offset):
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    #TODO 对于增广的参数，待调整输入值
    if is_train:
        # 进行增广
        dataset = LaneDataset(csv_file, image_size, crop_offset,
                              transform=transforms.Compose([
                                  ImageAug(), DeformAug(), ScaleAug(), CutOut(32, 0.5), ToTensor()]))
        data_batch = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    else:
        # 不进行增广
        dataset = LaneDataset(csv_file, image_size, crop_offset, transform=transforms.Compose([ToTensor()]))
        data_batch = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

    return data_batch



if __name__ == '__main__':
    data_batch = batch_data_generator()
    dataprocess = tqdm(data_batch)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(), mask.cuda()
        # this is aimed that debug your new method
        print(image.size(),mask.size(),type(image))
        import numpy as np
        import matplotlib.pyplot as plt
        image = image.cpu().numpy()
        print(type(image))
        plt.imshow(np.transpose(image[0],(1,2,0)))
        plt.show()
        mask = mask.cpu().numpy()
        plt.imshow(mask[0],cmap='gray_r')
        plt.show()
        break




