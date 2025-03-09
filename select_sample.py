import numpy as np
import torch
import os
from models.construct import model_construct

def distribute_budget(budget, n):
    avg_budget = budget / n
    lower_budget = int(np.floor(avg_budget))
    upper_budget = int(np.ceil(avg_budget))

    num_upper = budget - lower_budget * n  
    num_lower = n - num_upper 
    budget_distribution = [lower_budget] * num_lower + [upper_budget] * num_upper
    
    return budget_distribution

def entropy(probabilities):
    normalized_prob = probabilities / torch.sum(probabilities, dim=1, keepdim=True)
    entropy = -torch.sum(normalized_prob * torch.log2(normalized_prob), dim=1)
    return entropy

def sort(data, args, idx_train, idx_val, device):
    model = model_construct(args, args.model, data, device).to(device) 
    model.fit(data.x, data.edge_index, None, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=False)
    model.eval()
    
    logits = model(data.x, data.edge_index)
    
    unlabeled_idx = (torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.train_mask)).nonzero().flatten()
    
    full_label = torch.zeros_like(data.y)
    full_label[idx_train] = data.y[idx_train]
    full_label[unlabeled_idx] = torch.argmax(logits[unlabeled_idx], dim=1)
    
    entropy_score = entropy(logits).detach()
    
    sorted_unlabeled_idx = unlabeled_idx[entropy_score[unlabeled_idx].argsort(descending=True)]
    
    num_classes = int(full_label.max()) + 1
    
    sorted_samples = torch.empty(0, dtype=torch.long, device=device)
    class_pointers = {i: 0 for i in range(num_classes)}
    
    while len(sorted_samples) < len(unlabeled_idx):
        all_classes_exhausted = True
        for i in range(num_classes):
            class_idx = (full_label[sorted_unlabeled_idx] == i).nonzero(as_tuple=True)[0]
            
            if class_pointers[i] < len(class_idx):
                sorted_samples = torch.cat([sorted_samples, sorted_unlabeled_idx[class_idx[class_pointers[i]]].unsqueeze(0)])
                class_pointers[i] += 1
                all_classes_exhausted = False 
            
            if len(sorted_samples) >= len(unlabeled_idx):
                break
        
        if all_classes_exhausted:
            break
    
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, "sorted_samples")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, f"{args.dataset}/sorted_samples.pt")
    
    torch.save(sorted_samples, file_path)
    
    return sorted_samples

def select(data, args, idx_train, idx_val, device):
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, "sorted_samples")
    file_path = os.path.join(save_dir, f"{args.dataset}/sorted_samples.pt")

    if os.path.exists(file_path):
        print(f"Loading selected samples from: {file_path}")
        selected_samples = torch.load(file_path, map_location=device)
    else:
        print("sorted_samples.pt not found, calling sort() to generate it.")
        selected_samples = sort(data, args, idx_train, idx_val, device)
    budget = args.vs_number
    selected_samples = selected_samples[:budget]
    return selected_samples    