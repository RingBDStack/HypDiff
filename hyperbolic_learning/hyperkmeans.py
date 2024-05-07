# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
import sys
import os
import torch
# import modules within repository

from utils import *
from .hyperbolic_kmeans.hkmeans import HyperbolicKMeans, plot_clusters
# ignore warnings
import warnings
warnings.filterwarnings('ignore');
# # display multiple outputs within a cell
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all";

def hkmeanscom(args):
# load polbooks data

# load pre-trained embedding coordinates
    save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
    # if args.save:
    #     if not args.save_dir:
    #         # dt = datetime.datetime.now()
    #         # date = f"{dt.year}_{dt.month}_{dt.day}"
    #         # models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
    #         # save_dir = get_dir_name(models_dir)
    #         save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
    #     else:
    #         save_dir = args.save_dir
    #file_path = os.path.join('/home/wyc/Code/HyperDiff/logs/lp/2023_10_25/9/', 'embeddings.npy')
    file_path=os.path.join(save_dir, 'embeddings.npy')
    emb_data= np.load(file_path, allow_pickle=True)
    # print(emb_data.shape)
    #emb_data=torch.randn(105,2)
    # fit unsupervised clustering
    
    m=3
    hkmeans = HyperbolicKMeans(n_clusters=m)
    hkmeans.fit(emb_data, max_epochs=5)
    labels=hkmeans.assignments
    center=hkmeans.centroids
    lable_tmp=[]
    for i in range(len(labels)):
        for j in range(m):
            if(labels[i][j]==1):
                lable_tmp.append(j)
         
    #print(label_tmp.shape)
    # print(center.shape)
    lable_path=os.path.join(save_dir,'label.npy')
    center_path=os.path.join(save_dir,'center.npy')
    np.save(lable_path,lable_tmp)
    np.save(center_path,center)

def graph_hkmeanscom(args, emb_data, dataloader):

    save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
    # if args.save:
    #     if not args.save_dir:
    #         # dt = datetime.datetime.now()
    #         # date = f"{dt.year}_{dt.month}_{dt.day}"
    #         # models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
    #         # save_dir = get_dir_name(models_dir)
    #         save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
    #     else:
    #         save_dir = args.save_dir
    #file_path = os.path.join('/home/wyc/Code/HyperDiff/logs/lp/2023_10_25/9/', 'embeddings.npy')
    # file_path=os.path.join(save_dir, 'embeddings.npy')
    # emb_data= np.load(file_path, allow_pickle=True)
    # print(emb_data.shape)
    #emb_data=torch.randn(105,2)
    # fit unsupervised clustering
    # emb_data = torch.tensor(emb_data).float()
    emb_data = emb_data.numpy()
    label=[]
    centers=[]
    # for batch_idx, data in enumerate(dataloader):
    for batch_idx in range(len(emb_data)):
        m=3
        hkmeans = HyperbolicKMeans(n_clusters=m)
        hkmeans.fit(emb_data[batch_idx], max_epochs=5)
        labels=hkmeans.assignments
        center=hkmeans.centroids
        lable_tmp=[]
        for i in range(len(labels)):
            for j in range(m):
                if(labels[i][j]==1):
                    lable_tmp.append(j)
        label.append(lable_tmp)
        centers.append(center)
        #print(label_tmp.shape)
        # print(center.shape)
        lable_path=os.path.join(save_dir,'label'+str(batch_idx)+'.npy')
        center_path=os.path.join(save_dir,'center'+str(batch_idx)+'.npy')
        np.save(lable_path,lable_tmp)
        np.save(center_path,center)
    lable_path = os.path.join(save_dir, 'label' + '.npy')
    center_path = os.path.join(save_dir, 'center' + '.npy')
    np.save(lable_path,label)
    np.save(center_path,centers)