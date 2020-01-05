import torch
import torch.nn as nn
import numpy as np

#TODO 根据逻辑，统一predict和label的输入维度，免去每一次都转化维度的麻烦！！！参考代码中，维度

# predict由模型产出的Tensor，label由cv读取灰度图片并转化为Tensor
# predict:维度NCHW。先转化NHWC在变成N*H*W, C； label维度NHW，直接转成N*H*W, 1
def CrossEntropyLoss(predict, label, num_classes):
    predict = predict.permute(0, 2, 3, 1) # predict结果维度是NCHW，由NCHW转化为NHWC，用于reshape
    predict = torch.reshape(predict.contiguous(),[-1, num_classes])
    label = label.contiguous().view([-1, 1])
    loss = nn.CrossEntropyLoss(reduction='mean')(predict, label)
    return loss

#TODO 待定！！！
# 输入predict，label是torch。predict:维度NCHW。先转化NHWC在变成N*H*W, C； label维度NHW，直接转成N*H*W, 1
def DiceLoss(predict, label, num_classes, smooth=1):
    assert predict.shape[0] == label.shape[0]
    predict = predict.permute(0, 2, 3, 1) # predict结果维度是NCHW，由NCHW转化为NHWC，用于reshape
    predict = torch.reshape(predict.contiguous(),[-1, num_classes])
    label = label.contiguous().view([-1, 1])
    # 分子 2xy
    num = 2*torch.sum(torch.mul(predict, label), dim=1) + smooth
    # 分母 x^2+y^2
    den = torch.sum(predict.pow(2) + label.pow(2), dim=1) + smooth
    # dice loss = 1- (2xy)/(x^2+y^2)
    loss = 1 - num.numpy()/den.numpy()
    return loss

# predict由torch.softmax产出的Tensor，label由cv读取灰度图片并转化为Tensor。
# predict维度是N,H,W，label的维度是N,H,W
# result用于记录一个epoch所有数据的总和。
def mean_iou(predict, label, result):
    predict = predict.numpy()
    label = label.numpy()
    for i in range(len(result)):
        predict_class_i = predict==i
        label_class_i = label==i
        # 对于第i类，predict和label都是正的pixel
        tp_i = np.sum(predict_class_i * label_class_i)
        # 对于第i类，predict为正 + label为正 - 两者重叠的区域
        ta_i = np.sum(predict_class_i) + np.sum(label_class_i) - tp_i
        # 记录当前batch下，第i类的累计值
        result['TP'][i] += tp_i
        result['TA'][i] += ta_i
    return result
