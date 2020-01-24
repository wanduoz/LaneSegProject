import os
import cv2
import torch
import config
import argparse
import pandas as pd
import numpy as np
from tqdm import trange
from models.UNet import UNet
from models.deeplabv3p import deeplabv3p
from utils.predict_fn import load_model, img_transform, get_prediction, expand_resize_color_data

parser = argparse.ArgumentParser(description='LaneSeg predict parameters')
# netword
parser.add_argument('-m', '--model', default='deeplabv3p,unet_resnet18', type=str)
# valid/test
parser.add_argument('-d', '--data', default="val", type=str)
# num of predict
parser.add_argument('-s', '--size', default="all", type=str)
# write origin image
parser.add_argument('-a', '--add', default=False, type=bool)
# return
args = parser.parse_args()

use_model = 'model_Last.pth.tar'

def main():
    # 数据大小
    size = None if args.size == "all" else int(args.size)
    # 定义预测的数据集
    if args.data == 'val':
        df = pd.read_csv(config.val_list_dir, nrows=size)
    elif args.data == 'test':
        df = pd.read_csv(config.test_list_dir, nrows=size)
    else:
        raise Exception("data list {} is not support".format(args.data))

    # 对每一张图片，做下面循环
    for i in trange(df.shape[0]):

        image_path, label_path = df['image'][i], df['label'][i]
        pred = torch.zeros((1, config.num_classes, config.IMG_SIZE[1], config.IMG_SIZE[0]), dtype=torch.float32) # 1, 8,384,1024
        if torch.cuda.is_available():
            pred = pred.cuda()

        # 对每一个模型，做下面的预测
        models = args.model.split(",")
        # 读取图片，写入，方便读取
        image, label = cv2.imread(image_path), cv2.imread(label_path)
        if args.add:
            cv2.imwrite(os.path.join(config.ensemble_path, "{}_origin_image.jpg".format(i)), image)
            cv2.imwrite(os.path.join(config.ensemble_path, "{}_origin_label.jpg".format(i)), label)

        for model_name in models:
            # 获取模型名称。
            if model_name == 'deeplabv3p':
                model = deeplabv3p(config.num_classes)
                save_model_path = os.path.join(config.save_model_path, model_name)
                print("\n ensemble result from model: {},  use backbone: {}".format("deeplabv3p", "Xception"))
            elif model_name.startswith("unet"):
                backbone = model_name.split("_")[1]
                model = UNet(backbone, config.num_classes) # 里面已经判断是否存在不对劲的backbone名称
                save_model_path = os.path.join(config.save_model_path, model_name)
                print("\n ensemble result from model: {},  use backbone: {}".format("unet", backbone))
            else:
                raise Exception("unvalid model name {}".format(model_name) )

            print("load model from path: {}".format(save_model_path))


            model_path = os.path.join(save_model_path, use_model)  # model_0010.pth.tar
            model = load_model(model, model_path)

            img = img_transform(image)
            pred += (model(img)) # 1,num_cls,h,w

        # 所有模型加载完成，进行求均值，做预测
        pred = pred/(len(models)) # 1, CLASS, H, W
        prediction = get_prediction(pred) # H, W
        color_mask = expand_resize_color_data(prediction, config.SUBMISSION_SIZE, config.crop_offset) # H, W -> 3, H_ori, W_ori
        cv2.imwrite(os.path.join(config.ensemble_path, "{}_predict_figure.jpg".format(i)), color_mask)


if __name__ == "__main__":
    main()