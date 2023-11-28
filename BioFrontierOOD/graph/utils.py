import numpy as np
from sklearn.metrics import accuracy_score
import torch

def compute_metrics(output, cls_labels, threshold):
    output_array = np.array(output)
    binary = (output_array >= threshold).astype(int)
    acc = accuracy_score(binary, cls_labels)
    return acc, acc

def transfer_to_device(batch, device):
    return [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]

def print_result(epoch, train_loss, val_loss, val_acc):
    print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
