
import os
import time
import config
import numpy as np
import torch
import torch.nn.functional as F
from models.deeplabv3p import deeplabv3p
from utils.data_feeder import train_image_gen

# 计算Mean Iou。输入pred维度是N*H*W,num_class, Label是N*H*W, 1
def mean_iou(pred, label, num_classes=8):
    pred = torch.argmax(pred, axis=1) #输入pred维度： N*H*W, Num_class
    pred = pred.int()
    label = label.int()
    miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes)
    return miou

# 计算损失
no_grad_set = []
def create_loss(predict, label, num_classes):
    predict = predict.permute(0, 2, 3, 1) # predict结果维度是NCHW，由NCHW转化为NHWC，用于reshape
    predict = torch.reshape(predict,[-1, num_classes])
    predict = F.softmax(predict,dim=1)
    label = label.view(-1, 1) # label读取结果是 NCHW但是C是1，所以不转化
    # BCE with DICE
    bce_loss = F.cross_entropy(predict, label)
    # dice_loss = fluid.layers.dice_loss(predict, label)
    no_grad_set.append(label.name)
    loss = bce_loss + dice_loss
    miou = mean_iou(predict, label, num_classes)
    return fluid.layers.reduce_mean(loss), miou

def create_network(train_image, train_label, classes, network='deeplabv3p', image_size=(1024, 384), for_test=False):
    if network == 'deeplabv3p':
        predict = deeplabv3p(train_image, classes)
    else:
        raise Exception('Not support this model:', network)
    print('The program will run', network)

    if for_test == False:
        loss, miou = create_loss(predict, train_label, classes)
        return loss, miou, predict
    elif for_test == True:
        return predict
    else:
        raise Exception('Wrong Status:', for_test)

# The main method
def main():
    IMG_SIZE =[1536, 512]
    SUBMISSION_SIZE = [3384, 1710]
    add_num = 13
    batch_size = 2
    log_iters = 100
    base_lr = 0.0006
    num_classes = 8
    save_model_iters = 2000
    network = 'deeplabv3p'
    save_model_path = os.path.join(config.save_model_path, network)
    use_pretrained = True
    epoches = 2
    crop_offset = 690
    train_list = config.train_list_dir


    iter_id = 0
    total_loss = 0.0
    total_miou = 0.0
    prev_time = time.time()


    train_reader = train_image_gen(train_list, batch_size, IMG_SIZE, crop_offset)
    # Create model and define optimizer
    reduced_loss, miou, pred = create_network(images, labels, num_classes, network=network, image_size=(IMG_SIZE[1], IMG_SIZE[0]), for_test=False)
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=base_lr)
    optimizer.minimize(reduced_loss, no_grad_set=no_grad_set)

    # Whether load pretrained model
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    if use_pretrained == True:
        fluid.io.load_params(exe, model_path)
        print("loaded model from: %s" % model_path)
    else:
        print("Train from initialized model.")

    # Parallel Executor to use multi-GPUs
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.allow_op_delay = True
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
    train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=reduced_loss.name,
                                       build_strategy=build_strategy, exec_strategy=exec_strategy)

    # Training
    for epoch in range(epoches):
        print('Start Training Epoch: %d'%(epoch + 1))
        train_length = len(train_list)
        for iteration in range(int(train_length / batch_size)):
            train_data = next(train_reader)
            results = train_exe.run(
                feed=get_feeder_data(train_data, place),
                fetch_list=[reduced_loss.name, miou.name])
            iter_id += 1
            total_loss += np.mean(results[0])
            total_miou += np.mean(results[1])

            if iter_id % log_iters == 0: # Print log
                end_time = time.time()
                print(
                "Iter - %d: train loss: %.3f, mean iou: %.3f, time cost: %.3f s"
                % (iter_id, total_loss / log_iters, total_miou / log_iters, end_time - prev_time))
                total_loss = 0.0
                total_miou = 0.0
                prev_time = time.time()

            if iter_id % save_model_iters == 0: # save model
                dir_name =save_model_path + str(epoch + add_num) + '_' + str(iter_id)
                fluid.io.save_params(exe, dirname=dir_name)
                print("Saved checkpoint: %s" % (dir_name))
        iter_id = 0
        dir_name = save_model_path + str(epoch + add_num) + '_end'
        fluid.io.save_params(exe, dirname=dir_name)
        print("Saved checkpoint: %s" % (dir_name))

# Main
if __name__ == "__main__":
    main()
