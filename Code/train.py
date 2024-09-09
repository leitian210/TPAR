import os
import argparse
import torch
import numpy as np
import load_data_int
import load_data_ext
import base_model_int
import base_model_ext
from utils import select_gpu

parser = argparse.ArgumentParser(description="Parser for RED-GNN")
parser.add_argument('--data_path', type=str, default='data/ICEWS14_int/')
parser.add_argument('--seed', type=str, default=1234)
parser.add_argument('--task', type=str, default='ext')  # 'ext' for extrapolation, 'int' for 'interpolation'

args = parser.parse_args()

task = None
if args.task == 'int':
    DataLoader = load_data_int.DataLoader
    BaseModel = base_model_int.BaseModel
elif args.task == 'int':
    DataLoader = load_data_ext.DataLoader
    BaseModel = base_model_ext.BaseModel
else:
    print('Task Undefined!')

class Options(object):
    pass

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
   
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')

    gpu = select_gpu()
    torch.cuda.set_device(gpu)
    print('gpu:', gpu)

    loader = DataLoader(args.data_path)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel

    if dataset == 'ICEWS14_int':
        opts.lr = 0.0003
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 128
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.2
        opts.act = 'idd'
        opts.n_batch = 10
        opts.n_tbatch = 10


    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    model = BaseModel(opts, loader)

    best_mrr = 0
    for epoch in range(50):
        mrr, out_str = model.train_batch()
        with open(opts.perf_file, 'a+') as f:
            f.write(out_str)
        if mrr > best_mrr:
            best_mrr = mrr
            best_str = out_str
            print(str(epoch) + '\t' + best_str)
    print(best_str)

