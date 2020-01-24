import cv2
import torch
import config
import numpy as np
from utils.image_process import crop_resize_data
from utils.label_process import decode_color_labels, decode_labels

device_id = 0

# 读取模型
def load_model(model, model_path):
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda(device=device_id)
        map_location = 'cuda:%d' % device_id
    else:
        map_location = 'cpu'
    #
    model_param = torch.load(model_path, map_location=map_location)['state_dict']
    model_param = {k.replace('module.', ''):v for k, v in model_param.items()}
    # net.load_state_dict(model_param.state_dict())
    model.load_state_dict(model_param)
    return model

# cv读取图片，经过data generator相同的处理，后放入模型得到pred
def img_transform(img):
    img = crop_resize_data(img,config.IMG_SIZE,config.crop_offset)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...].astype(np.float32)
    img = torch.from_numpy(img.copy())
    if torch.cuda.is_available():
        img = img.cuda(device=device_id)
    return img

# 由pred，通过argmax获取预测的class，numpy格式
def get_prediction(pred):
    pred = torch.softmax(pred, dim=1)
    pred_heatmap = torch.max(pred, dim=1)
    # 1,H,W,C
    pred = torch.argmax(pred, dim=1)
    pred = torch.squeeze(pred)
    pred = pred.detach().cpu().numpy()
    return pred

# 预测值扩充会原图大小（彩色标签）
def expand_resize_color_data(prediction, submission_size, offset):
    color_pred_mask = decode_color_labels(prediction) # H,W -> 3, H, W
    color_pred_mask = np.transpose(color_pred_mask, (1, 2, 0)) # 3,H,W -> H,W,3
    color_expand_mask = cv2.resize(color_pred_mask, (submission_size[0], submission_size[1] - offset), interpolation=cv2.INTER_NEAREST)
    color_submission_mask = np.zeros((submission_size[1], submission_size[0], 3), dtype='uint8')
    color_submission_mask[offset:, :, :] = color_expand_mask
    return color_submission_mask

# 灰度标签
def expand_resize_data(prediction, submission_size, offset):
    pred_mask = decode_labels(prediction)
    expand_mask = cv2.resize(pred_mask, (submission_size[0], submission_size[1] - offset), interpolation=cv2.INTER_NEAREST)
    submission_mask = np.zeros((submission_size[1], submission_size[0]), dtype='uint8')
    submission_mask[offset:, :] = expand_mask
    return submission_mask