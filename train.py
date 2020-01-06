
import os
import torch
import config
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from models.deeplabv3p import deeplabv3p
from utils.loss import CrossEntropyLoss, mean_iou
from utils.data_feeder import batch_data_generator

# 训练一个epoch的函数
def train_epoch(model, epoch, dataLoader, optimizer, trainLog):
    model.train()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        image = image.type(torch.FloatTensor)
        if torch.cuda.is_available():
            image, mask = image.cuda(), mask.cuda()
        optimizer.zero_grad()
        out = model(image) # N, NUM_CLS, H, W
        mask_loss = CrossEntropyLoss(out, mask, config.num_classes) # 返回tensor。不能返回标量，因为要backward
        total_mask_loss += mask_loss.item()
        mask_loss.backward()
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))

    trainLog.write("Epoch:{:04d}, mask loss is {:.4f} \n".format(epoch, total_mask_loss/len(dataLoader)))
    trainLog.flush()


# 验证一个epoch的函数
def val_epoch(model, epoch, dataLoader, valLog):
    model.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    result = {"TP":{i:0 for i in range(config.num_classes)}, "TA":{i:0 for i in range(config.num_classes)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        image = image.type(torch.FloatTensor)
        if torch.cuda.is_available():
            image, mask = image.cuda(), mask.cuda()
        out = model(image)
        mask_loss = CrossEntropyLoss(out, mask, config.num_classes)
        total_mask_loss += mask_loss.detach().item()
        predict = torch.argmax(F.softmax(out,dim=1), dim=1) # N C H W降到 N H W
        result = mean_iou(predict, mask, result)
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask loss:{:.4f}".format(mask_loss))

    valLog.write("Epoch:{}".format(epoch))
    for i in range(config.num_classes):
        result_string = "{}: {:.4f} \n".format(i, result["TP"]/result["TA"])
        valLog.write(result_string)

    valLog.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss/len(dataLoader)))
    valLog.flush()


def parse_args():
    parser = argparse.ArgumentParser(description='LaneSeg input parameters')
    # batch_size
    parser.add_argument('-bs','--batchsize',default=4,type=int)
    # netword
    parser.add_argument('-n','--model',default='deeplabv3p',type=str)
    # epoch
    parser.add_argument('-e','--epoch',default=10,type=int)
    # return
    args = parser.parse_args()
    return args


def main(args):

    # 地址与记录loss的文件
    save_model_path = os.path.join(config.save_model_path, args.model)
    trainLog= open(os.path.join(save_model_path,'train.log'), 'w')
    valLog = open(os.path.join(save_model_path,'val.log'), 'w')

    # 获取batch data的generator
    train_data_batch = batch_data_generator(config.train_list_dir,
                                            is_train=True,
                                            batch_size=args.batchsize,
                                            image_size=config.IMG_SIZE,
                                            crop_offset=config.crop_offset)
    val_data_batch = batch_data_generator(config.val_list_dir,
                                          is_train=False,
                                          batch_size=args.batchsize,
                                          image_size=config.IMG_SIZE,
                                          crop_offset=config.crop_offset)

    # model
    if args.model == 'deeplabv3p':
        model = deeplabv3p(config.num_classes)
    else:
        raise Exception("model {} is not supported".format(args.model))

    if torch.cuda.is_available():
        model = model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr, weight_decay=config.weight_decay)

    for epoch in range(args.epoch):
        train_epoch(model, epoch, train_data_batch, optimizer, trainLog, save_model_path, )
        val_epoch(model, epoch, val_data_batch, valLog)
        if epoch % 10 == 0:
            torch.save(model, os.path.join(save_model_path, "model_{:04d}.pth".format(epoch) ) )
    trainLog.close()
    valLog.close()
    torch.save(model, os.path.join(save_model_path, "model_Last.pth" ) )

# Main
if __name__ == "__main__":
    args = parse_args()
    main(args)
