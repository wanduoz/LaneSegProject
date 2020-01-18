import os
import torch
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_model_path, patience, epsilon):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            epsilon (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_model_path = save_model_path
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.epsilon = epsilon

    def __call__(self, val_loss, model, epoch):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)

        elif score < self.best_score + self.epsilon:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 先删除原有的early stop保存
        for i in os.listdir(self.save_model_path):
            if i.startswith("early_stop"):
                os.remove(i)
        # 再保存新的early stop保存点
        torch.save({'state_dict': model.state_dict()}, os.path.join(self.save_model_path, "early_stop{:04d}.pth.tar".format(epoch)))
        self.val_loss_min = val_loss

