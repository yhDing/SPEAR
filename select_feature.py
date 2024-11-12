import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.construct import model_construct
from torch_geometric.datasets import Planetoid,Flickr,Amazon

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Pubmed','ogbn-arxiv'])
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='MLP', help='model',
                    choices=['GCN','MLP'])
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--device_id', type=int, default=3,
                    help="Threshold of prunning edges")
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--sample_num', type=int, default=16,
                    help='Number of samples in sage.')

args = parser.parse_known_args()[0]
args.cuda =  torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/', \
                    transform=transform)
elif(args.dataset == 'Photo'):
    dataset = Amazon(root='./data/', \
                     name='Photo', \
                    transform=transform)
elif(args.dataset == 'Computers'):
    dataset = Amazon(root='./data/', \
                     name='Computers', \
                    transform=transform)

elif(args.dataset == 'ogbn-arxiv'):
    from ogb.nodeproppred import PygNodePropPredDataset
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 

data = dataset[0].to(device)

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
# we build our own train test split 
#%% 
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

import sage_modified as sage

pre_train = model_construct(args,args.model,data,device).to(device)
pre_train.fit(data.x, data.y, idx_train, idx_val, train_iters=args.epochs,verbose=False)

x, y = data.x.cpu().numpy(), data.y
num_classes = torch.max(y) + 1
y = torch.nn.functional.one_hot(y, num_classes=num_classes).cpu().numpy()
feature_names = [str(i) for i in range(0, data.x.shape[1])]

model = pre_train
imputer = sage.MarginalImputer(model, x[:args.sample_num])
estimator = sage.PermutationEstimator(imputer, 'mse')
sage_values = estimator(x, y)
val, std = sage_values.save_num()

np.save(f'save_selected_feature/{args.dataset}/val_{args.sample_num}.npy', val)
np.save(f'save_selected_feature/{args.dataset}/std_{args.sample_num}.npy', std)

figure = sage_values.plot(feature_names,return_fig=True)

figure.savefig("saved_plot.png", dpi=600, bbox_inches='tight')

plt.show()

