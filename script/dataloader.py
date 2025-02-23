import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pdb

def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()
    
    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228

    return adj, n_vertex

def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    return train, val, test

def data_transform(data, n_his, n_pred, n_vertex, device="cuda:0"):
    len_record = len(data)
    num = len_record - n_his - n_pred

    x = torch.zeros([num, 1, n_his, n_vertex])
    y = torch.zeros([num, n_vertex])

    chunk_size = 1000
    for chunk_start in range(0, num, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num)
        
        for i in range(chunk_start, chunk_end):
            head = i
            tail = i + n_his
            x[i, :, :, :] = torch.from_numpy(data[head: tail].reshape(1, n_his, n_vertex))
            y[i] = torch.from_numpy(data[tail + n_pred - 1])

    x = x.pin_memory()
    y = y.pin_memory()

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    return x, y