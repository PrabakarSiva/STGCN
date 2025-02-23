import math
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, delta: float = 0.0, patience: int = 7, verbose: bool = True, path: str = '/media/data2/ITS/STGCN/checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'           
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = math.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            with open('val_losses.txt', 'w') as f:
                f.write(f'{val_loss:.6f}\n')
        elif score < self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            with open('val_losses.txt', 'a') as f:
                f.write(f'{val_loss:.6f}\n')
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            with open('val_losses.txt', 'a') as f:
                f.write(f'{val_loss:.6f}\n')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if val_loss < self.val_loss_min:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
