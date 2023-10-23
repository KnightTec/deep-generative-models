import torch

def likelihood(X_train, model, device):
    ##########################################################
    # YOUR CODE HERE

    ##########################################################
    X_train = X_train.to(device)
    batch_log_prob = model.log_prob(X_train)
    loss = -batch_log_prob.mean(dim=0)
    return loss
