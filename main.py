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
    from graph_train import train
    #  an easy encoder: encoding from adj
    if args.type=='train': 
        save_dir = os.path.join(os.environ['LOG_DIR'], args.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # f2=open('logs/timecp/' +args.dataset +'_time.txt', 'a+') 
        # f2.flush()
        # t_total = time.time()

        embeddings, dataloader = train(args)

        # Hyperbolic Geometric Diffusion Process
        hyperdiff_graphset(args, embeddings, dataloader)
        # hyperdiff_graphset(args)

        # t_end = time.time()
        # f2.write('{0:4} {1:4} \n'.format('time',t_end - t_total))
        # f2.close()

    elif args.type=='test':
        test_graphset(args)
