import pandas as pd
from methods import *
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
import torch
from sklearn import model_selection 
from scipy import stats
from __init__ import train_neural_net, draw_neural_net


data = pd.read_csv("data.csv")

# Transform famhist into numerical values 
X, y, attributeNames = transformData(data)

N, M = X.shape
partARegression = False
# Add offset attribute
if partARegression:
       X = np.concatenate((np.ones((X.shape[0],1)),X),1)
       attributeNames = np.insert(attributeNames,0,'Offset')
       M = M+1
       ## Crossvalidation
       K = 10
       # Values of lambda
       lambdas = np.power(10.,range(-4,8))
       opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas,K)
       figure(2,figsize=(12,8))
       subplot(1,2,1)
       semilogx(lambdas, mean_w_vs_lambda.T[:,1:],'.-',
              [opt_lambda]*len(np.linspace(-0.22,0.8,40)),np.linspace(-0.22,0.8,40), 'k-')
       legend(attributeNames[1:])
       grid()
       xlabel("Regularization factor")
       ylabel("Mean coefficient value")
       subplot(1,2,2)
       loglog(lambdas,train_err_vs_lambda.T,'b.-',
              lambdas,test_err_vs_lambda.T,'r.-',
              [opt_lambda]*len(np.linspace(3.39,4.3,40)),np.linspace(3.39,4.3,40), 'g-')
       legend(['Train error','Generalization error',"Optimal lambda: 1e{0}".format(np.log10(opt_lambda))],loc='upper left')
       xlabel('Regularization factor')
       ylabel('Squared error (cross validation)')
       show()

########################################### PART B ##################################################
RegressionBLambda = True
if RegressionBLambda == True:    
       # Add offset attribute
       X = np.concatenate((np.ones((X.shape[0],1)),X),1)
       attributeNames = np.insert(attributeNames,0,'Offset')
       M = M+1

       ## Crossvalidation
       # Create crossvalidation partition for evaluation
       K = 2
       CV = model_selection.KFold(K, shuffle=True)
       #CV = model_selection.KFold(K, shuffle=False)

       # Values of lambda
       lambdas = np.power(10.,range(-4,8))

       # Initialize variables
       T = len(lambdas)
       Error_train = np.empty((K,1))
       Error_test = np.empty((K,1))
       Error_train_rlr = np.empty((K,1))
       Error_test_rlr = np.empty((K,1))
       Error_train_nofeatures = np.empty((K,1))
       Error_test_nofeatures = np.empty((K,1))
       w_rlr = np.empty((M,K))
       mu = np.empty((K, M-1))
       sigma = np.empty((K, M-1))
       w_noreg = np.empty((M,K))
       opt_lambdas = []

       k=0
       for train_index, test_index in CV.split(X,y):
       
              # extract training and test set for current CV fold
              X_train = X[train_index]
              y_train = y[train_index]
              X_test = X[test_index]
              y_test = y[test_index]
              internal_cross_validation = 2
              
              opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
              opt_lambdas.append(opt_lambda)
              # Standardize outer fold based on training set, and save the mean and standard
              # deviations since they're part of the model (they would be needed for
              # making new predictions) - for brevity we won't always store these in the scripts
              mu[k, :] = np.mean(X_train[:, 1:], 0)
              sigma[k, :] = np.std(X_train[:, 1:], 0)
              X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
              X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
              Xty = X_train.T @ y_train
              XtX = X_train.T @ X_train
              
              # Compute mean squared error without using the input data at all
              Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
              Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

              # Estimate weights for the optimal value of lambda, on entire training set
              lambdaI = opt_lambda * np.eye(M)
              lambdaI[0,0] = 0 # Do no regularize the bias term
              w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
              # Compute mean squared error with regularization with optimal lambda
              Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
              Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

              # Estimate weights for unregularized linear regression, on entire training set
              w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
              # Compute mean squared error without regularization
              Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
              Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]

              # To inspect the used indices, use these print statements
              k+=1

       print("Linear Regression")
       print("Error in each fold")
       for index,val in enumerate(Error_test_rlr):
              print("Fold {0}: {1}".format(index+1,val[0]))
       print("Mean: " + str(np.mean(Error_test_rlr)))

       print("Baseline")
       print("Error in each fold")
       for index,val in enumerate(Error_test_nofeatures):
              print("Fold {0}: {1}".format(index+1,val[0]))
       print("Mean: " + str(np.mean(Error_test_nofeatures)))
              
       print("Optimal lambda for each fold")

       for index,val in enumerate(opt_lambdas):
              print("Fold {0}: {1}".format(index+1,val))
       print("Mean: " + str(np.mean(opt_lambdas)))
       

       

TwoLayerANN = False
if TwoLayerANN:
       # Parameters for neural network classifier
       n_hidden_units = 2      # number of hidden units
       n_replicates = 1       # number of networks trained in each k-fold
       max_iter = 10000
       y = y.reshape((N,1))
       # K-fold crossvalidation
       K1 = 10                # only three folds to speed up this example
       CV1 = model_selection.KFold(K1, shuffle=True)
       K2 = 10
       hs = [1, 2, 3, 4, 5]
       best_hs = []
       errors = []

       loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

       for (k, (train_index, test_index)) in enumerate(CV1.split(X, y)):
              print('\nOutter crossvalidation fold: {0}/{1}'.format(k+1, K1))
              X_train = torch.Tensor(X[train_index, :])
              y_train = torch.Tensor(y[train_index])
              X_test = torch.Tensor(X[test_index, :])
              y_test = torch.Tensor(y[test_index])
              best_loss_matrix = [[] for _ in range(len(hs))]
              generalization_error = [0 for _ in range(len(hs))]
              CV2 = model_selection.KFold(K2,shuffle=True)
              for (k1, (train_index_IF, test_index_IF)) in enumerate(CV2.split(X_train,y_train)):
                     X_train_IF = torch.Tensor(X[train_index_IF, :])
                     y_train_IF = torch.Tensor(y[train_index_IF])
                     X_test_IF = torch.Tensor(X[test_index_IF, :])
                     y_test_IF = torch.Tensor(y[test_index_IF])
                     print('\nInner crossvalidation fold: {0}/{1}'.format(k1+1, K2))
                     for val in hs:
                            n_hidden_units = val
                            def model(): return torch.nn.Sequential(torch.nn.Linear(
                            M, n_hidden_units), torch.nn.Tanh(), torch.nn.Linear(n_hidden_units, 1),)
                            # Train the net on training data
                            net, final_loss, learning_curve = train_neural_net(model,
                                                                             loss_fn,
                                                                             X=X_train_IF,
                                                                             y=y_train_IF,
                                                                             n_replicates=n_replicates,
                                                                             max_iter=max_iter)
                            best_loss_matrix[val-1].append((final_loss))
              for j in range(len(hs)):
                     for final_loss in best_loss_matrix[j]:
                            generalization_error[j] += final_loss
                     generalization_error[j] *= ((N/K1-(N/(K1*K2)))/(N/K1))*(K2)
              best_h = np.argmin(generalization_error) + 1
              n_hidden_units = best_h
              def model(): return torch.nn.Sequential(torch.nn.Linear(
              M, n_hidden_units), torch.nn.Tanh(), torch.nn.Linear(n_hidden_units, 1),)
              # Train the net on training data
              net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_IF,
                                                               y=y_train_IF,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
              y_est_test = net(X_test)
              se = (y_est_test.float() - y_test.float())**2
              mse = (sum(se).type(torch.float)/len(y_test)).data.numpy()
              best_hs.append(best_h)
              errors.append(mse[0])
              print("Best h in the outer fold {0}={1} with the generalization error msE_gen={2}".format(k+1,best_h,mse[0]))
              
                     
       print("Best hidden units with corresponding MSE are: ")
       for i in range(len(best_hs)):
              print("Best h in the outer fold {0}={1} with the generalization error E_gen={2}".format(i+1,best_hs[i],errors[i]))
