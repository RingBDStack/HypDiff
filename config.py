import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.001, 'learning rate'),
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
        'dataset': ('CL100', 'which dataset to use'),
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
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
