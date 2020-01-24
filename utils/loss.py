import torch
import config
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# 根据逻辑，统一predict和label的输入维度，免去每一次都转化维度的麻烦！！！参考代码中，维度
# predict由模型产出的Tensor，label由cv读取灰度图片并转化为Tensor
# predict:维度NCHW。先转化NHWC在变成N*H*W, C； label维度NHW，直接转成N*H*W, 1
def CrossEntropyLoss(predict, label, num_classes, weight=None):
    predict = predict.permute(0, 2, 3, 1) # predict结果维度是NCHW，由NCHW转化为NHWC，用于reshape
    predict = predict.contiguous().view(-1,num_classes)
    label = label.contiguous().view(-1)
    if weight is not None:
        if torch.cuda.is_available():
            weight = torch.FloatTensor(weight).cuda(device='cuda:0')
        else:
            weight = torch.FloatTensor(weight)
    loss = nn.CrossEntropyLoss(weight=weight, reduction='mean')(predict, label)
    return loss
################################################################################


################################################################################
# tensor 变成 onehot
# 输入input维度N,1,*(*表示后面随意）。输出维度N,CLS,*
def make_one_hot(input, num_classes):
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    if torch.cuda.is_available():
        result = result.cuda()
    return result

# 类似multi label的ovr
# predict/label输入都是第i类  predict的维度：N,H,W, label的维度：N,H,W
def BinaryDiceLoss(predict, label):
    assert predict.shape[0] == label.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1)
    label = label.contiguous().view(label.shape[0], -1)
    num = 2 * torch.sum(torch.mul(predict, label), dim=1) + 1
    den = torch.sum(predict.pow(2) + label.pow(2), dim=1) + 1
    loss = 1 - num / den
    return loss.mean()

# dice loss使用参考代码。
# 输入的predict，label维度和ce loss一样
# predict由模型产出的Tensor，label由cv读取灰度图片并转化为Tensor
# predict:维度NCHW。 label维度NHW
# weight输入(CLS,)的列表
def DiceLoss(predict, label, num_class, weight=None):
    label = label.unsqueeze(1) # N,H,W -> N,1,H,W
    label = make_one_hot(label, num_class) # N,1,H,W -> N,CLS,H,W
    assert predict.shape == label.shape, 'predict & target shape do not match'

    total_loss = 0
    predict = F.softmax(predict, dim=1)
    if weight is not None:
        if torch.cuda.is_available():
            weight = torch.FloatTensor(weight).cuda(device='cuda:0')
        else:
            weight = torch.FloatTensor(weight)
        assert weight.shape[0] == label.shape[1], 'Expect weight shape [{}], get[{}]'.format(label.shape[1], weight.shape[0])
    # 对于每一个类别。label的第一个维度是CLS的。
    for i in range(label.shape[1]):
        dice_loss = BinaryDiceLoss(predict[:, i], label[:, i])
        if weight is not None:
            dice_loss *= weight[i]
        total_loss += dice_loss
    assert isinstance(total_loss,torch.Tensor), "dice loss is not torch tensor"
    return total_loss/label.shape[1]
################################################################################

# predict由torch.softmax产出的Tensor，label由cv读取灰度图片并转化为Tensor。
# predict维度是N,H,W，label的维度是N,H,W
# result用于记录一个epoch所有数据的总和。
def mean_iou(predict, label, result):
    predict = predict.cpu().numpy()
    label = label.cpu().numpy()
    for i in range(config.num_classes):
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
