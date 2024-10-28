# #%%
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import utils
# from copy import deepcopy
# from torch_geometric.nn import GCNConv
# import numpy as np
# import scipy.sparse as sp
# from torch_geometric.utils import from_scipy_sparse_matrix
# from torch_geometric.utils import subgraph

# class GCN(nn.Module):

#     def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False):

#         super(GCN, self).__init__()

#         assert device is not None, "Please specify 'device'!"
#         self.device = device
#         self.nfeat = nfeat
#         self.hidden_sizes = [nhid]
#         self.nclass = nclass
#         self.convs = nn.ModuleList()
#         self.convs.append(GCNConv(nfeat, nhid))
#         self.lns = nn.ModuleList()
#         self.lns.append(torch.nn.LayerNorm(nfeat))
#         for _ in range(layer-2):
#             self.convs.append(GCNConv(nhid,nhid))
#             self.lns.append(nn.LayerNorm(nhid))
#         self.lns.append(nn.LayerNorm(nhid))
#         self.gc2 = GCNConv(nhid, nclass)
#         self.dropout = dropout
#         self.lr = lr
#         self.output = None
#         self.edge_index = None
#         self.edge_weight = None
#         self.features = None 
#         self.weight_decay = weight_decay

#         self.layer_norm_first = layer_norm_first
#         self.use_ln = use_ln

#     def forward(self, x, edge_index, edge_weight=None):
#         if(self.layer_norm_first):
#             x = self.lns[0](x)
#         i=0
#         for conv in self.convs:
#             x = F.relu(conv(x, edge_index,edge_weight))
#             if self.use_ln:
#                 x = self.lns[i+1](x)
#             i+=1
#             x = F.dropout(x, self.dropout, training=self.training)
#         features = x
#         # print('features',features)
#         x = self.gc2(x, edge_index,edge_weight)
#         return F.log_softmax(x,dim=1)
#     def get_h(self, x, edge_index):

#         for conv in self.convs:
#             x = F.relu(conv(x, edge_index))
        
#         return x

#     def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, finetune=False, attach=None):
#         """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
#         Parameters
#         ----------
#         features :
#             node features
#         adj :
#             the adjacency matrix. The format could be torch.tensor or scipy matrix
#         labels :
#             node labels
#         idx_train :
#             node training indices
#         idx_val :
#             node validation indices. If not given (None), GCN training process will not adpot early stopping
#         train_iters : int
#             number of training epochs
#         initialize : bool
#             whether to initialize parameters before training
#         verbose : bool
#             whether to show verbose logs

#                 """



#         self.edge_index, self.edge_weight = edge_index, edge_weight
#         self.features = features.to(self.device)
#         self.labels = labels.to(self.device)

#         if idx_val is None:
#             self._train_without_val(self.labels, idx_train, train_iters, verbose)
#         else:
#             if finetune==True:
#                 self.fientune(self.labels, idx_train, idx_val, attach, train_iters, verbose)
#             else:
#                 self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
#         # torch.cuda.empty_cache()

#     def fientune(self, labels, idx_train, idx_val, idx_attach, train_iters, verbose):
#         # idx1 = idx_train[:-len(idx_attach)]
#         # idx2 = idx_train[-len(idx_attach):]
#         # idx1 = [item for item in idx_train if item not in idx_attach]

#         idx_train_set = set(idx_train)
#         idx_attach_set = set(idx_attach)
#         idx1 = list(idx_train_set - idx_attach_set)
#         idx2 = idx_attach

#         idx1 = torch.tensor(idx1).to(self.device)
#         idx2 = torch.tensor(idx2).to(self.device)
        

#         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

#         for i in range(train_iters):
#             self.train()
#             optimizer.zero_grad()
#             output = self.forward(self.features, self.edge_index, self.edge_weight)
#             loss_train = F.nll_loss(output[idx1], labels[idx1])
#             probs = F.softmax(output[idx2], dim=1)
#             target_probs = probs[range(len(labels[idx2])), labels[idx2]]
#             loss_train_2 = torch.mean(target_probs)  # Mean of probabilities of correct labels

#             # Combining the normal and adversarial losses
#             loss_train = loss_train + loss_train_2
#             loss_train.backward()
#             optimizer.step()
#             # if i % 10 == 0:
#             #     print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
    
#     # def finetune(self, labels, idx_train, idx_val, idx_attach, train_iters, verbose):
#         # Prepare index sets for training
#         # idx_train_set = set(idx_train)
#         # idx_attach_set = set(idx_attach)
#         # idx1 = list(idx_train_set - idx_attach_set)
#         # idx2 = list(idx_attach_set)

#         # # Extract subgraph for the combined indices
#         # idx_subgraph = list(set(idx1 + idx2))
#         # subgraph_edge_index, subgraph_edge_weight = subgraph(
#         #     subset=idx_subgraph,
#         #     edge_index=self.edge_index,
#         #     edge_attr=self.edge_weight,
#         #     relabel_nodes=True,
#         #     num_nodes=self.features.size(0),
#         # )

#         # # Convert lists to tensor and map to device
#         # idx1 = torch.tensor([idx_subgraph.index(x) for x in idx1]).to(self.device)
#         # idx2 = torch.tensor([idx_subgraph.index(x) for x in idx2]).to(self.device)

#         # # Create optimizer
#         # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

#         # # Training iterations
#         # for i in range(train_iters):
#         #     self.train()
#         #     optimizer.zero_grad()

#         #     # Forward pass on the subgraph
#         #     output, x = self.forward(
#         #         self.features[idx_subgraph].to(self.device),
#         #         subgraph_edge_index,
#         #         subgraph_edge_weight
#         #     )

#         #     # Calculate loss on idx1 (training nodes not in idx_attach)
#         #     loss_train = F.nll_loss(output[idx1], labels[idx1])

#         #     # Calculate the probability loss for idx2 (training nodes in idx_attach)
#         #     probs = F.softmax(output[idx2], dim=1)
#         #     target_probs = probs[torch.arange(len(labels[idx2])), labels[idx2]]
#         #     loss_train_2 = torch.mean(target_probs)  # Mean of probabilities of correct labels

#         #     # Combine losses and backpropagate
#         #     loss_train = loss_train + loss_train_2
#         #     loss_train.backward()
#         #     optimizer.step()

#         #     # if verbose and i % 10 == 0:
#         #     if i % 10 == 0:
#         #         print('Epoch {}, training loss: {:.4f}'.format(i, loss_train.item()))

#         self.eval()
#             # output, x = self.forward(self.features, self.edge_index, self.edge_weight)
#             # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#             # acc_val = utils.accuracy(output[idx_val], labels[idx_val])
#             # print("acc_val: {:.4f}".format(acc_val))
#             # if acc_val > best_acc_val:
#             #     best_acc_val = acc_val
#         self.output = output

#     def _train_without_val(self, labels, idx_train, train_iters, verbose):
#         self.train()
#         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         for i in range(train_iters):
#             optimizer.zero_grad()
#             output = self.forward(self.features, self.edge_index, self.edge_weight)
#             loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            
#             loss_train.backward()
#             optimizer.step()
#             if verbose and i % 10 == 0:
#                 print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

#         self.eval()
#         output = self.forward(self.features, self.edge_index, self.edge_weight)
#         self.output = output
#         # torch.cuda.empty_cache()

#     def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
#         if verbose:
#             print('=== training gcn model ===')
#         optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

#         best_loss_val = 100
#         best_acc_val = 0

#         for i in range(train_iters):
#             self.train()
#             optimizer.zero_grad()
#             output = self.forward(self.features, self.edge_index, self.edge_weight)
#             loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#             # print(labels[idx_train])
#             loss_train.backward()
#             optimizer.step()



#             self.eval()
#             output = self.forward(self.features, self.edge_index, self.edge_weight)
#             loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#             acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
#             # if verbose and i % 10 == 0:
#             # print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
#             # print("acc_val: {:.4f}".format(acc_val))
#             if acc_val > best_acc_val:
#                 best_acc_val = acc_val
#                 self.output = output
#                 # weights = deepcopy(self.state_dict())

#         if verbose:
#             print('=== picking the best model according to the performance on validation ===')
#         # self.load_state_dict(weights)
#         # torch.cuda.empty_cache()


#     def test(self, features, edge_index, edge_weight, labels,idx_test):
#         """Evaluate GCN performance on test set.
#         Parameters
#         ----------
#         idx_test :
#             node testing indices
#         """
#         self.eval()
#         # print(labels[idx_test])
#         with torch.no_grad():
#             output = self.forward(features, edge_index, edge_weight)
#             # print(torch.exp(output[idx_test]))
#             # print(output[idx_test].max(1)[1])
#             acc_test = utils.accuracy(output[idx_test], labels[idx_test])
#         # torch.cuda.empty_cache()
#         # print("Test set results:",
#         #       "loss= {:.4f}".format(loss_test.item()),
#         #       "accuracy= {:.4f}".format(acc_test.item()))
#         return float(acc_test)
    
#     def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
#         self.eval()
#         output = self.forward(features, edge_index, edge_weight)
#         correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
#         acc_test = utils.accuracy(output[idx_test], labels[idx_test])
#         # torch.cuda.empty_cache()
#         return acc_test,correct_nids

# # %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.utils import subgraph

class GCN_loop(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False,add_self_loops=True):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.add_self_loops = add_self_loops
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, add_self_loops=self.add_self_loops))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer-2):
            self.convs.append(GCNConv(nhid,nhid, add_self_loops=self.add_self_loops))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, nclass, add_self_loops=self.add_self_loops)
        # print('add_selfloop',self.gc2.add_self_loops)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay

        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

    def forward(self, x, edge_index, edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        features = x
        # print('features',features)
        x = self.gc2(x, edge_index,edge_weight)
        return F.log_softmax(x,dim=1)
    def get_h(self, x, edge_index):

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        return x

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, finetune=False, attach=None):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs

                """



        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            if finetune==True:
                self.fientune(self.labels, idx_train, idx_val, attach, train_iters, verbose)
            else:
                self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
        # torch.cuda.empty_cache()

    def fientune(self, labels, idx_train, idx_val, idx_attach, train_iters, verbose):
        # idx1 = idx_train[:-len(idx_attach)]
        # idx2 = idx_train[-len(idx_attach):]
        # idx1 = [item for item in idx_train if item not in idx_attach]

        idx_train_set = set(idx_train)
        idx_attach_set = set(idx_attach)
        idx1 = list(idx_train_set - idx_attach_set)
        idx2 = idx_attach

        idx1 = torch.tensor(idx1).to(self.device)
        idx2 = torch.tensor(idx2).to(self.device)
        

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx1], labels[idx1])
            probs = F.softmax(output[idx2], dim=1)
            target_probs = probs[range(len(labels[idx2])), labels[idx2]]
            loss_train_2 = torch.mean(target_probs)  # Mean of probabilities of correct labels

            # Combining the normal and adversarial losses
            loss_train = loss_train + loss_train_2
            loss_train.backward()
            optimizer.step()
            # if i % 10 == 0:
            #     print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
    
    # def finetune(self, labels, idx_train, idx_val, idx_attach, train_iters, verbose):
        # Prepare index sets for training
        # idx_train_set = set(idx_train)
        # idx_attach_set = set(idx_attach)
        # idx1 = list(idx_train_set - idx_attach_set)
        # idx2 = list(idx_attach_set)

        # # Extract subgraph for the combined indices
        # idx_subgraph = list(set(idx1 + idx2))
        # subgraph_edge_index, subgraph_edge_weight = subgraph(
        #     subset=idx_subgraph,
        #     edge_index=self.edge_index,
        #     edge_attr=self.edge_weight,
        #     relabel_nodes=True,
        #     num_nodes=self.features.size(0),
        # )

        # # Convert lists to tensor and map to device
        # idx1 = torch.tensor([idx_subgraph.index(x) for x in idx1]).to(self.device)
        # idx2 = torch.tensor([idx_subgraph.index(x) for x in idx2]).to(self.device)

        # # Create optimizer
        # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # # Training iterations
        # for i in range(train_iters):
        #     self.train()
        #     optimizer.zero_grad()

        #     # Forward pass on the subgraph
        #     output, x = self.forward(
        #         self.features[idx_subgraph].to(self.device),
        #         subgraph_edge_index,
        #         subgraph_edge_weight
        #     )

        #     # Calculate loss on idx1 (training nodes not in idx_attach)
        #     loss_train = F.nll_loss(output[idx1], labels[idx1])

        #     # Calculate the probability loss for idx2 (training nodes in idx_attach)
        #     probs = F.softmax(output[idx2], dim=1)
        #     target_probs = probs[torch.arange(len(labels[idx2])), labels[idx2]]
        #     loss_train_2 = torch.mean(target_probs)  # Mean of probabilities of correct labels

        #     # Combine losses and backpropagate
        #     loss_train = loss_train + loss_train_2
        #     loss_train.backward()
        #     optimizer.step()

        #     # if verbose and i % 10 == 0:
        #     if i % 10 == 0:
        #         print('Epoch {}, training loss: {:.4f}'.format(i, loss_train.item()))

        self.eval()
            # output, x = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            # acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            # print("acc_val: {:.4f}".format(acc_val))
            # if acc_val > best_acc_val:
            #     best_acc_val = acc_val
        self.output = output

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output
        # torch.cuda.empty_cache()

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # print(labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            # if verbose and i % 10 == 0:
            # print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
            # print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                # weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        # self.load_state_dict(weights)
        # torch.cuda.empty_cache()


    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        # print(labels[idx_test])
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            # print(torch.exp(output[idx_test]))
            # print(output[idx_test].max(1)[1])
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        return acc_test,correct_nids

# %%

