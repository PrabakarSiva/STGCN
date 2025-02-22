import logging
import os
import gc
import argparse
import math
import random
import warnings
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping, opt
from model import models
import networkx as nx
from torch_geometric.utils import to_dense_adj
import pdb
#import nni

# Global device variable
device = None

def set_env(seed):
    # Set available CUDA devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def get_parameters():
    global device
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=3, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'nadamw', 'lion'], help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
        torch.cuda.empty_cache() # Clean cache
    else:
        device = torch.device('cpu')
        gc.collect() # Clean cache
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    return args, device, blocks

def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args.dataset)
    print(f'NUMBER OF VERTICES: {n_vertex}')
    gso = utility.calc_gso(adj, args.gso_type)
    gso_coo = gso.tocoo()
    edge_index = torch.tensor([gso_coo.row, gso_coo.col], dtype=torch.long).to(device)

    print(edge_index.max())
    print(edge_index.shape)
    print(n_vertex)

    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]

    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, args.n_his, args.n_pred, n_vertex, device)
    x_val, y_val = dataloader.data_transform(val, args.n_his, args.n_pred, n_vertex, device)
    x_test, y_test = dataloader.data_transform(test, args.n_his, args.n_pred, n_vertex, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter, edge_index

def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    path_to_save = f'/media/data2/ITS/STGCN/STGCN_{args.dataset}.pt'
    es = earlystopping.EarlyStopping(delta=0.0, 
                                     patience=args.patience, 
                                     verbose=True, 
                                     path=path_to_save)

    model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    if args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "nadamw":
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, decoupled_weight_decay=True)
    elif args.opt == "lion":
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler

def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter, edge_index):
    lambda_ge = 0.1
    max_grad_norm = 5.0
    
    train_loss_file = open('train_losses.txt', 'w')
    val_loss_file = open('val_losses.txt', 'w')
    
    print("Starting training...")

    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0
        model.train()
        epoch_adj_preds = []
        epoch_main_loss = 0.0
        n = 0

        true_adj = to_dense_adj(edge_index)[0].cpu().numpy()
        G_true = nx.from_numpy_array(true_adj)
        del true_adj
        
        for x, y in tqdm.tqdm(train_iter, desc=f'Epoch {epoch+1}/{args.epochs}'):
            x, y = x.to(device), y.to(device)  # Ensure data is on correct device
            optimizer.zero_grad()
            y_pred, adj_pred = model(x, edge_index)
            
            main_loss = loss(y_pred.view(len(x), -1), y)

            epoch_adj_preds.append(adj_pred.mean(dim=1).detach().cpu())
            
            main_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            epoch_main_loss += main_loss.item() * y.shape[0]
            n += y.shape[0]

            del y_pred, adj_pred, main_loss
            torch.cuda.empty_cache()

        all_adj_preds = torch.cat(epoch_adj_preds, dim=0)
        avg_adj_pred = torch.mean(all_adj_preds, dim=0).numpy()
        G_pred = nx.from_numpy_array(avg_adj_pred)
        del avg_adj_pred
        
        ged = nx.graph_edit_distance(G_pred, G_true)
        ge_loss = next(ged)
        
        ge_loss_tensor = torch.tensor(ge_loss, device=device, requires_grad=True)
        (lambda_ge * ge_loss_tensor).backward()
        optimizer.step()
        
        l_sum = epoch_main_loss + lambda_ge * ge_loss

        del epoch_adj_preds, G_pred, G_true, ge_loss_tensor
        torch.cuda.empty_cache()
        
        train_loss = l_sum / n
        print(f'Epoch Loss: Main = {epoch_main_loss/n:.6f}, Graph Edit = {ge_loss:.6f}, Total = {train_loss:.6f}')
        
        scheduler.step()
        val_loss = val(model, edge_index, val_iter, loss)
        print(f'Epoch: {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        
        train_loss_file.write(f'Epoch {epoch+1}: {train_loss:.6f}\n')
        val_loss_file.write(f'Epoch {epoch+1}: {val_loss:.6f}\n')

        es(val_loss, model)
        if es.early_stop:
            print("Early stopping")
            train_loss_file.close()
            val_loss_file.close()
            break

    train_loss_file.close()
    val_loss_file.close()


@torch.no_grad()
def val(model, edge_index, val_iter, loss):
    model.eval()
    print("\nRunning validation...")

    l_sum, n = 0.0, 0
    for x, y in tqdm.tqdm(val_iter, desc='Validating'):
        x, y = x.to(device), y.to(device)
        y_pred, _ = model(x, edge_index)
        y_pred = y_pred.view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n, device=device)

@torch.no_grad() 
def test(zscore, loss, model, test_iter, edge_index, args):
    model.load_state_dict(torch.load(f"STGCN_{args.dataset}.pt"))
    model.eval()
    
    test_results_file = open('test_results.txt', 'w')
    
    print("\nTesting model...")
    test_MSE = utility.evaluate_model(model, loss, test_iter, edge_index)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore, edge_index)
    
    result_str = f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}'
    print(result_str)
    test_results_file.write(result_str + '\n')
    test_results_file.close()

if __name__ == "__main__":
    # Logging
    logging.basicConfig(level=logging.INFO)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter, edge_index = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
    train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter, edge_index)
    test(zscore, loss, model, test_iter, edge_index, args)
