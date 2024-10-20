import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_lp_config': {
        'lr': (0.0001, 'learning rate'),
        'dropout': (0.2, 'dropout probability'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (10, 'maximum number of epochs to train for'),
        'weight-decay': (0.00001, 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'momentum': (0.99, 'momentum in optimizer'),
        'seed': (1432, 'seed for training'),
        'log-freq': (5, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.2, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (10, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (None, 'do not early stop before min-epochs')
    },
    'model_config': {
        'model': ('HGCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN]'),
        'dim': (512, 'embedding dimension'),
        'hid1': (64, 'embedding dimension'),
        'hid2': (32, 'embedding dimension'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Hyperboloid, PoincareBall]'),
        'c': (0.6, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (0, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.6, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'dataset': ('MUTAG', 'which dataset to use'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (0, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1432, 'seed for data splits (train/test/val)'),
    },
    'work_type_config':{
        'type':('train','which type to choose, you can select train or test'), 
        'diff_epoc':(1, 'maximum number of epochs to train for'),
        'taskselect':('graphtask', '[lptask, graphtask]')
    },
    'training_diffusion_config': {
        'target': ('pred_noise', 'at every Vis_step, the plots will be updated'),
        'restrict': (False, 'whether use the geometric contraints'),
        'lr_diff': (0.0001, 'model learning rate'),
        'epoch_diff': (5000, 'maximum number of epochs to train for'),
        'epoch_load': (5000, 'maximum number of epochs to train for')
    },
    'training_hvae_config': {
        'Vis_step': (1000, 'at every Vis_step, the plots will be updated'),
        'redraw': (False, 'either update the log plot each step'),
        'epoch_number': (5000, 'maximum number of epochs to train for'),
        'graphEmDim': (64, 'the dimention of graph Embeding LAyer; z'),
        'graph_save_path': (None, 'the direc to save generated synthatic graphs'),
        'use_feature': (True, 'either use features or identity matrix'),
        'PATH': ('model', 'a string which determine the path in wich model will be saved'),
        'decoder': ('FC', 'the decoder type, FC is only option in this rep'),
        'encoder_type': ("HAvePool", 'the encoder: only option in this rep is Ave'),
        'batchSize': (200, 'the size of each batch; the number of graphs is the mini batch'),
        'UseGPU': (True, 'either use GPU or not if availabel'),
        'model_vae': ('graphVAE', 'only option is graphVAE'),
        'device': ("cuda:0", 'Which device should be used'),
        'task': ("graphGeneration", 'only option in this rep is graphGeneration'),
        'bfsOrdering': (True, 'use bfs for graph permutations'),
        'directed': (True, 'is the dataset directed?!'),
        'beta': (None, 'beta coefiicieny'),
        'plot_testGraphs': (True, 'shall the test set be printed'),
        'ideal_Evalaution': (False, 'if you want to comapre the 50%50 subset of dataset comparision?!')
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
