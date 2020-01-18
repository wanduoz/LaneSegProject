import os
import torch
import config
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from models.deeplabv3p import deeplabv3p
from models.UNet import UNet
from utils.loss import CrossEntropyLoss, mean_iou
from utils.data_feeder import batch_data_generator
from utils.early_stop import EarlyStopping

parser = argparse.ArgumentParser(description='LaneSeg input parameters')
# batch_size
parser.add_argument('-b', '--batchsize', default=4, type=int)
# netword
parser.add_argument('-m', '--model', default='deeplabv3p', type=str)
# epoch
parser.add_argument('-e', '--epoch', default=21, type=int)
# cuda
parser.add_argument('-c','--cuda', default="0",type=str)
# Unet backbone name (resnet18/resnet34/resnet50/resnet101/resnet152)
parser.add_argument('-n', '--backbone', default="resnet18",type=str)
# return
args = parser.parse_args()

# 默认0。可以输入多个比如"0,1,2"
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
print("use gpu: ", torch.cuda.device_count(), "GPUs!\t", "use_gpu: ",args.cuda)

# 记录每一个iteration下的loss（不是每一个epoch）
ce_loss_train = []
ce_loss_val = []

# 训练一个epoch的函数
def train_epoch(model, epoch, dataLoader, optimizer, trainLog):

    model.train()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        image = image.type(torch.FloatTensor)
        mask = mask.type(torch.LongTensor)
        if torch.cuda.is_available():
            image, mask = image.cuda(device='cuda:0'), mask.cuda(device='cuda:0')
            # image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        optimizer.zero_grad()
        out = model(image) # N, NUM_CLS, H, W
        mask_loss = CrossEntropyLoss(out, mask, config.num_classes) # 返回tensor。不能返回标量，因为要backward

        total_mask_loss += mask_loss.item()
        ce_loss_train.append(mask_loss.item())

        mask_loss.backward()
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))

    curr_train_avg_loss = total_mask_loss/len(dataLoader)
    trainLog.write("Epoch:{:04d}, mask loss is {:.4f} \n".format(epoch, curr_train_avg_loss))
    trainLog.flush()

    return curr_train_avg_loss


# 验证一个epoch的函数
def val_epoch(model, epoch, dataLoader, valLog):

    model.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    result = {"TP":{i:0 for i in range(config.num_classes)}, "TA":{i:0 for i in range(config.num_classes)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        image = image.type(torch.FloatTensor)
        mask = mask.type(torch.LongTensor)
        if torch.cuda.is_available():
            image, mask = image.cuda(device='cuda:0'), mask.cuda(device='cuda:0')
            # image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        with torch.no_grad():
            out = model(image)
            mask_loss = CrossEntropyLoss(out, mask, config.num_classes)

            total_mask_loss += mask_loss.detach().item()
            ce_loss_val.append(mask_loss.detach().item())

            predict = torch.argmax(F.softmax(out,dim=1), dim=1) # N C H W降到 N H W
            result = mean_iou(predict, mask, result)
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask loss:{:.4f}".format(mask_loss))

    miou_value = 0
    for i in range(config.num_classes):
        result_string = "class: {}, {:.4f} {:.4f};  miou is: {:.4f}\n".format(i, result["TP"][i],
                                                                              result["TA"][i], result["TP"][i]/result["TA"][i],)
        print(result_string)
        if i!=0:
            miou_value += result["TP"][i]/result["TA"][i]

    curr_val_avg_loss = total_mask_loss / len(dataLoader)
    valLog.write("Epoch:{}, miou is {:.4f} \n".format(epoch, miou_value/(config.num_classes-1)))
    valLog.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, curr_val_avg_loss))
    valLog.flush()

    return curr_val_avg_loss

def main():

    # 地址与记录loss的文件
    save_model_path = os.path.join(config.save_model_path, args.model)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

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

    # 定义模型model
    if args.model == 'deeplabv3p':
        model = deeplabv3p(config.num_classes)
    elif args.model == 'unet':
        model = UNet(args.name, config.num_classes)
    else:
        raise Exception("model {} is not supported".format(args.model))
    print("use model: {}".format(args.model))

    if torch.cuda.is_available():
        model = model.cuda()
        if len(args.gpus.split(',')) > 1:
            model = torch.nn.DataParallel(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr, weight_decay=config.weight_decay)

    # 早停。save_model_path, patience, epsilon
    early_stopping = EarlyStopping(save_model_path, config.patience, config.epsilon)

    # 开始训练
    # 每一个epoch，进行train，val，early_stop，判断是否需要早停。保存模型（每5epoch）
    for epoch in range(args.epoch):
        train_loss = train_epoch(model, epoch, train_data_batch, optimizer, trainLog)
        val_loss = val_epoch(model, epoch, val_data_batch, valLog)

        if epoch % 5 == 0:
            torch.save({'state_dict': model.state_dict()}, os.path.join(save_model_path, "model_{:04d}.pth.tar".format(epoch) ) )

        # 打印当前batch的avg loss信息
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    trainLog.close()
    valLog.close()
    torch.save({'state_dict': model.state_dict()}, os.path.join(save_model_path, "model_Last.pth.tar" ) )

    pd.DataFrame(ce_loss_train,columns=['loss']).to_csv(os.path.join(save_model_path,'ce_loss_train.csv'))
    pd.DataFrame(ce_loss_val, columns=['loss']).to_csv(os.path.join(save_model_path, 'ce_loss_val.csv'))

# Main
if __name__ == "__main__":
    main()
