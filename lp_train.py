from __future__ import division
from __future__ import print_function

import json
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import LPModel
from utils.data_utils import load_data
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import scipy.sparse as sp
from tqdm import tqdm

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load data
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    
    args.n_nodes, args.feat_dim = data['features'].shape

    args.nb_false_edges = len(data['train_edges_false'])
    args.nb_edges = len(data['train_edges'])
    Model = LPModel
    # No validation for reconstruction task
    args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    model = Model(args)
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    for epoch in tqdm(range(args.epochs)):
        tt = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings,t,h0,adj= model.encode(data['features'], data['adj_train_norm'])
        # print(h0)

        train_metrics = model.compute_metrics(embeddings, data, t,h0,adj,'train')
        # print(train_metrics['loss'])
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings,t,h0,adj = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, t,h0,adj,'val')

    model.eval()

    best_test_metrics = model.compute_metrics(embeddings, data,t,h0, adj,'test')

    # print(best_test_metrics)
    print('End encoding!')
    np.save(os.path.join(save_dir, 'embeddings.npy'), h0.cpu().detach().numpy())
    if hasattr(model.encoder, 'att_adj'): 
        filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
        pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
        print('Dumped attention adj: ' + filename)

    json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
