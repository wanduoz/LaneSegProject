import config
import argparse
import numpy as np
from tqdm import tqdm
from utils.data_feeder import batch_data_generator

parser = argparse.ArgumentParser(description='LaneSeg predict parameters')
# batchsize
parser.add_argument('-b', '--batchsize', default=4, type=int)
# return
args = parser.parse_args()

def main():

    train_data_batch = batch_data_generator(config.train_list_dir,
                                            is_train=False,
                                            batch_size=args.batchsize,
                                            image_size=config.IMG_SIZE,
                                            crop_offset=config.crop_offset)


    number_class = {i: 0 for i in range(8)}
    train_data_batch = tqdm(train_data_batch)
    for item in train_data_batch:
        temp = item['mask'].numpy()
        for i in range(config.num_classes):
            number_class[i] += np.sum(temp==i)
    for i in range(config.num_classes):
        print("{} has number of {}".format(i, number_class[i]))


if __name__ == "__main__":
    main()
