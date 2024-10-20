from config import parser
from diff import hyperdiff,test,hyperdiff_graphset,test_graphset

import os
import time

args = parser.parse_args()
os.environ['DATAPATH'] = 'data/'
os.environ['LOG_DIR'] = 'logs/'+args.taskselect+'/'

if args.taskselect=='lptask':
    from lp_train import train
    if args.type=='train': 
        #  Hyperbolic Geometric Auto-encoding
        train(args) 
        # Hyperbolic Geometric Diffusion Process
        hyperdiff(args)

    elif args.type=='test':
        # test based on a trained model
        test(args) 

elif args.taskselect=='graphtask': 
    from graph_train import graph_train
    #  an easy encoder: encoding from adj
    graph_train(args)
    from diff_graphset import diff_train
    diff_train(args)