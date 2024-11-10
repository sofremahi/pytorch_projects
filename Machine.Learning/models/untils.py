import torch
def accuracy_fn(y_True , y_preds):
    correct = torch.eq(y_True , y_preds).summ().item()
    return (correct/len(y_preds))*100