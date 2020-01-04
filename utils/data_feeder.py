import torch
import config
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import ScaleAug, CutOut, ToTensor
from utils.image_process import LaneDataset, ImageAug, DeformAug


def train_image_gen(train_list, batch_size=4, image_size=(1024, 384), crop_offset=690):
    # kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    # 不进行增广
    training_dataset = LaneDataset(train_list, image_size, crop_offset, transform=transforms.Compose(ToTensor() ))
    # 进行增广
    # training_dataset = LaneDataset("train.csv", transform=transforms.Compose([ImageAug(), DeformAug(), ScaleAug(), CutOut(32,0.5), ToTensor()]))

    training_data_batch = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dataprocess = tqdm(training_data_batch)
    # 返回DataLoader对象
    return dataprocess


if __name__ == '__main__':
    dataprocess = train_image_gen()
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(), mask.cuda()
        # this is aimed that debug your new method
        print(image.size(),mask.size(),type(image))
        # image = image.numpy()
        # print(type(image))
        # plt.imshow(np.transpose(image[0],(1,2,0)))
        # plt.show()
        # plt.imshow(mask[0])
        # plt.show()




