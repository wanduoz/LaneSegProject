import os
import cv2
import config
import argparse
import pandas as pd
from tqdm import trange
from models.UNet import UNet
from models.deeplabv3p import deeplabv3p
from utils.predict_fn import load_model, img_transform, get_prediction, expand_resize_color_data

use_model = 'model_Last.pth.tar'

device_id = 0
# predict_net = 'deeplabv3p'
# nets = {'deeplabv3p': DeeplabV3Plus, 'unet': ResNetUNet}
parser = argparse.ArgumentParser(description='LaneSeg predict parameters')
# netword
parser.add_argument('-m', '--model', default='deeplabv3p', type=str)
# Unet backbone name (resnet18/resnet34/resnet50/resnet101/resnet152)
parser.add_argument('-n', '--backbone', default="resnet18",type=str)
# valid/test
parser.add_argument('-d', '--data', default="val", type=str)
# num of predict
parser.add_argument('-s', '--size', default="all", type=str)
# write origin image
parser.add_argument('-a', '--add', default=False, type=bool)
# return
args = parser.parse_args()


def main():

    # 定义模型model
    if args.model == 'deeplabv3p':
        model = deeplabv3p(config.num_classes)
        save_model_path = os.path.join(config.save_model_path, args.model)
        print("use model: {},  use backbone: {}".format(args.model, "Xception"))
    elif args.model == 'unet':
        model = UNet(args.backbone, config.num_classes)
        save_model_path = os.path.join(config.save_model_path, args.model + "_" + args.backbone)
        print("use model: {},  use backbone: {}".format(args.model, args.backbone))
    else:
        raise Exception("model {} is not supported".format(args.model))
    print("load model from path: {}".format(save_model_path))

    size = None if args.size == "all" else int(args.size)
    # 定义预测的数据集
    if args.data == 'val':
        df = pd.read_csv(config.val_list_dir, nrows=size)
    elif args.data == 'test':
        df = pd.read_csv(config.test_list_dir, nrows=size)
    else:
        raise Exception("data list {} is not support".format(args.data))


    for i in trange(df.shape[0]):

        model_path = os.path.join(save_model_path, use_model) # model_0010.pth.tar
        model = load_model(model, model_path)

        image_path, label_path = df['image'][i], df['label'][i]
        image, label = cv2.imread(image_path),  cv2.imread(label_path)
        if args.add:
            cv2.imwrite(os.path.join(config.pred_model_path, "{}_origin_image.jpg".format(i)), image)
            cv2.imwrite(os.path.join(config.pred_model_path, "{}_origin_label.jpg".format(i)), label)
        image = img_transform(image)
        pred = model(image)
        prediction = get_prediction(pred) # H,W
        color_mask = expand_resize_color_data(prediction, config.SUBMISSION_SIZE, config.crop_offset) # H,W -> 3,H,W -> H,W,3 -> H_ORI, W_PRO, 3
        cv2.imwrite(os.path.join(config.pred_model_path, "{}_predict_figure.jpg".format(i)), color_mask)

if __name__ == '__main__':
    main()
