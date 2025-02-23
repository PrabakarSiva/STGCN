import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch

def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    eigval_max = norm(gso, 2)

    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso

def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

def evaluate_model(model, loss, data_iter, edge_index):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred, _ = model(x, edge_index)
            y_pred = y_pred.view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def calculate_normalized_laplacian_batch(adj_batch):
    if adj_batch.ndim == 4:
        adj_batch = adj_batch.mean(axis=1)
    elif adj_batch.ndim == 2:
        adj_batch = adj_batch[np.newaxis, ...]
    elif adj_batch.ndim != 3:
        raise ValueError(f"Invalid adjacency matrix shape: {adj_batch.shape}")
    
    batch_size, n, n2 = adj_batch.shape
    assert n == n2, f"Adjacency matrix should be a square, got shape {adj_batch.shape}"

    identity = np.eye(n)

    degrees = np.sum(adj_batch, axis=2)
    degrees = np.maximum(degrees, np.ones_like(degrees) * 1e-10)
    deg_inv_sqrt = np.power(degrees, -0.5)

    deg_inv_sqrt = deg_inv_sqrt[..., np.newaxis]

    normalized_adj = adj_batch * deg_inv_sqrt * deg_inv_sqrt.transpose(0, 2, 1)
    normalized_laplacian = np.tile(identity, (batch_size, 1, 1)) - normalized_adj
    
    return normalized_laplacian

def calculate_frobenius_distance(adj1, adj2, gso_type='sym_norm_lap'):
    if torch.is_tensor(adj1):
        device = adj1.device
        adj1 = adj1.detach()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adj1 = torch.from_numpy(adj1).to(device)

    if not torch.is_tensor(adj2):
        adj2 = torch.from_numpy(adj2).to(device)

    degrees1 = adj1.sum(dim=-1)
    degrees1 = torch.clamp(degrees1, min=1e-10)
    deg_inv_sqrt1 = degrees1.pow(-0.5).unsqueeze(-1)

    if adj2.dim() <= 2:
        degrees2 = adj2.sum(dim=-1)
        degrees2 = torch.clamp(degrees2, min=1e-10)
        deg_inv_sqrt2 = degrees2.pow(-0.5).unsqueeze(-1)
        L2 = torch.eye(adj2.size(-1), device=device) - (deg_inv_sqrt2 * adj2 * deg_inv_sqrt2.transpose(-2, -1))
        L2 = L2.unsqueeze(0).expand(adj1.size(0), -1, -1)
    else:
        degrees2 = adj2.sum(dim=-1)
        degrees2 = torch.clamp(degrees2, min=1e-10)
        deg_inv_sqrt2 = degrees2.pow(-0.5).unsqueeze(-1)
        L2 = torch.eye(adj2.size(-1), device=device).unsqueeze(0) - (deg_inv_sqrt2 * adj2 * deg_inv_sqrt2.transpose(-2, -1))

    L1 = torch.eye(adj1.size(-1), device=device).unsqueeze(0) - (deg_inv_sqrt1 * adj1 * deg_inv_sqrt1.transpose(-2, -1))

    return torch.norm(L1 - L2, p='fro', dim=(1,2)).mean()

def evaluate_metric(model, data_iter, scaler, edge_index):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred, _ = model(x, edge_index)
            y_pred = scaler.inverse_transform(y_pred.view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        return MAE, RMSE, WMAPE
