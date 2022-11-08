
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
from sklearn import model_selection
from __init__ import rlr_validate, train_neural_net
import torch


def basicStatistics(data):
    """Return the mean, median, and standard deviation of the data."""
    mean = np.mean(data, axis=0)
    median = np.median(data, axis=0)
    std = np.std(data, axis=0)
    return mean, median, std


def normalizeData(data):
    """Normalize the data to zero mean."""
    mean, _, std = basicStatistics(data)
    data = (data - mean)/std
    return data

def transformData(data):
    """Convert text to numbers."""
    data['famhist'] = data['famhist'].map({'Present': 1, 'Absent': 0})
    data.drop('chd', axis=1, inplace=True)
    data.drop('row.names', axis=1, inplace=True)
    data.drop('famhist', axis=1, inplace=True)
    y = data['ldl'].squeeze()
    data.drop('ldl', axis=1, inplace=True)
    attributeNames = data.columns
    return np.array(data), np.array(y), np.array(attributeNames)

def BaselineCrossValidation(train_index, test_index, y):
    y_train = y[train_index]
    y_test = y[test_index]
    Error_test_nofeatures = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    return Error_test_nofeatures

def LinearRegressionCrossValidation(train_index, test_index, X, y, K_inner, lambdas):
    X = np.concatenate((np.ones((X.shape[0],1)),X),1)
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, K_inner)
    
    mu =  np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)
    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma 
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    lambdaI = opt_lambda * np.eye(X.shape[1])
    lambdaI[0] = 0
    w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    Error_test_rlr = np.square(y_test-X_test @ w_rlr).sum(axis=0)/y_test.shape[0]
    return  Error_test_rlr, opt_lambda

    

def ANNCrossValidation(train_index, test_index, X, y, hs, K_Inner, K_Outter):
    n_replicates = 1       # number of networks trained in each k-fold
    max_iter = 10000
    y = y.reshape((X.shape[0],1))
    loss_fn = torch.nn.MSELoss()
    X_train = torch.Tensor(X[train_index, :])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index, :])
    y_test = torch.Tensor(y[test_index])




    CV_inner = model_selection.KFold(K_Inner, shuffle=True, random_state=1234)
    Error_validation_IF = [[] for _ in range(len(hs))]

    for (k, (train_index_IF, test_index_IF)) in enumerate(CV_inner.split(X_train,y_train)):
        X_train_IF = torch.Tensor(X[train_index_IF, :])
        y_train_IF = torch.Tensor(y[train_index_IF])
        X_test_IF = torch.Tensor(X[test_index_IF, :])
        y_test_IF = torch.Tensor(y[test_index_IF])
        print('Inner crossvalidation fold: {0}/{1}'.format(k+1, K_Inner))
        for val in hs:
            n_hidden_units = val
            def model(): return torch.nn.Sequential(torch.nn.Linear(
            X.shape[1], n_hidden_units), torch.nn.Tanh(), torch.nn.Linear(n_hidden_units, 1),)
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                                loss_fn,
                                                                X=X_train_IF,
                                                                y=y_train_IF,
                                                                n_replicates=n_replicates,
                                                                max_iter=max_iter)
            
            y_test_est = net(X_test_IF)
            se = (y_test_est.float() - y_test_IF.float())**2
            mse = (sum(se).type(torch.float)/len(y_test_IF)).data.numpy()
            Error_validation_IF[val-1].append(mse)
    generalization_error = [0 for _ in range(len(hs))]
    for j in range(len(hs)):
        for final_loss in Error_validation_IF[j]:
            generalization_error[j] += final_loss
        generalization_error[j] *= ((X.shape[0]/K_Outter-(X.shape[0]/(K_Outter*K_Inner)))/(X.shape[0]/K_Outter))*(K_Inner)
    best_h = np.argmin(generalization_error) + 1
    n_hidden_units = best_h
    def model(): return torch.nn.Sequential(torch.nn.Linear(
    X.shape[1], n_hidden_units), torch.nn.Tanh(), torch.nn.Linear(n_hidden_units, 1),)
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                    loss_fn,
                                                    X=X_train,
                                                    y=y_train,
                                                    n_replicates=n_replicates,
                                                    max_iter=max_iter)
    y_test_est = net(X_test)
    se = (y_test_est.float() - y_test.float())**2
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy()

    return best_h, mse

def LinearRegressionFit(X_train, X_test, y_train, opt_lambda):
    X_train_LR = np.concatenate((np.ones((X_train.shape[0],1)),X_train),1)
    X_test_LR = np.concatenate((np.ones((X_test.shape[0],1)),X_test),1)
    lambdaI = opt_lambda * np.eye(X_train_LR.shape[1])
    lambdaI[0,0] = 0
    mu = np.mean(X_train_LR[:, 1:], 0)
    sigma = np.std(X_train_LR[:, 1:], 0) 
    X_train_LR[:, 1:] = (X_train_LR[:, 1:] - mu) / sigma 
    X_test_LR[:, 1:] = (X_test_LR[:, 1:] - mu) / sigma 
    Xty = X_train_LR.T @ y_train
    XtX = X_train_LR.T @ X_train_LR
    w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    y_est_test = X_test_LR @ w_rlr
    return y_est_test

def ANNFit(X_train, X_test, y_train, y_test, h):
    n_hidden_units = h
    n_replicates = 1
    max_iter = 10000
    y_train = y_train.reshape((X_train.shape[0],1))
    no_attributes = X_train.shape[1]
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    loss_fn = torch.nn.MSELoss()
    def model(): return torch.nn.Sequential(torch.nn.Linear(no_attributes, n_hidden_units), torch.nn.Tanh(), torch.nn.Linear(n_hidden_units, 1),)
    
    net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train,
                                                        y=y_train,
                                                        n_replicates=n_replicates,
                                                        max_iter=max_iter)
    
    y_est_test = net(X_test)
    return (y_est_test).detach().numpy()[0]

