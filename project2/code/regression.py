import pandas as pd
from methods import *
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend,
                              title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
import torch
from sklearn import model_selection


# Reading data
data = pd.read_csv("data.csv")

# Transform famhist into numerical values
X, y, attributeNames = transformData(data)

X = stats.zscore(X)
# print(X)
# y = normalizeData(y)

N, M = X.shape

partARegression = False
# Add offset attribute
if partARegression:
    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
    attributeNames = np.insert(attributeNames, 0, 'Offset')
    M = M+1
    # Crossvalidation
    K = 10
    # Values of lambda
    lambdas = np.power(10., range(-4, 8))
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(
        X, y, lambdas, K)
    figure(2, figsize=(12, 8))
    subplot(1, 2, 1)
    semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-',
             [opt_lambda]*len(np.linspace(-0.22, 0.8, 40)), np.linspace(-0.22, 0.8, 40), 'k-')
    legend(attributeNames[1:])
    grid()
    xlabel("Regularization factor")
    ylabel("Mean coefficient value")
    subplot(1, 2, 2)
    loglog(lambdas, train_err_vs_lambda.T, 'b.-',
           lambdas, test_err_vs_lambda.T, 'r.-',
           [opt_lambda]*len(np.linspace(3.39, 4.3, 40)), np.linspace(3.39, 4.3, 40), 'g-')
    legend(['Train error', 'Generalization error', "Optimal lambda: 1e{0}".format(
        np.log10(opt_lambda))], loc='upper left')
    xlabel('Regularization factor')
    ylabel('Squared error (cross validation)')
    show()

########################################### PART B ##################################################
CrossValidation = False
if CrossValidation:
    K_Outter = 10
    K_Inner = 10
    hs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    CV_Outter = model_selection.KFold(
        K_Outter, shuffle=True, random_state=1234)

    Error_test_baseline = [0 for _ in range(K_Outter)]
    Error_test_LR = [0 for _ in range(K_Outter)]
    best_lambda = [0 for _ in range(K_Outter)]
    best_hs = [0 for _ in range(K_Outter)]
    Error_test_ANN = [0 for _ in range(K_Outter)]

    for k, (train_index, test_index) in enumerate(CV_Outter.split(X, y)):

        print("Outter cross validation {0}/{1}".format(k+1, K_Outter))

        for i in range(3):
            if i == 0:
                print("Baseline")
                Error_test_baseline[k] = BaselineCrossValidation(
                    train_index, test_index, y)
                print("Inner cross validation done.\n")
            elif i == 1:
                print("Linear Regression")
                lambdas = np.power(10., range(-4, 8))
                Error_test_LR[k], best_lambda[k] = LinearRegressionCrossValidation(
                    train_index, test_index, X, y, K_Inner, lambdas)
                print("Inner cross validation done.\n")

            elif i == 2:
                print("ANN")
                best_hs[k], Error_test_ANN[k] = ANNCrossValidation(
                    train_index, test_index, X, y, hs, K_Inner, K_Outter)
                print("Inner cross validation done.\n")

    print("Summary:")

    print("==================================================================")
    print("Baseline:")
    for index, val in enumerate(Error_test_baseline):
        print("Fold {0}: {1}".format(index+1, val))
    print("Mean: " + str(np.mean(Error_test_baseline)))
    print("==================================================================\n")

    print("==================================================================")
    print("Linear Regression")
    for index, val in enumerate(Error_test_LR):
        print("Fold {0}: Error {1}".format(index+1, val))
    print("Mean: " + str(np.mean(Error_test_LR)))
    print("Best Lambdas")
    for index, val in enumerate(best_lambda):
        print("Fold {0}: Error {1}".format(index+1, val))
    print("Mean: " + str(np.mean(best_lambda)))
    print("==================================================================\n")

    print("==================================================================")
    print("Hidden units with corresponding MSE are: ")
    for i in range(len(Error_test_ANN)):
        print("Best h in the outter fold {0} = {1} with the generalization error E_gen={2}".format(
            i+1, best_hs[i], Error_test_ANN[i][0]))
    print("Mean Error: {0}".format(np.mean(Error_test_ANN)))
    print("==================================================================")


h = 1
opt_lambda = 100

Statistics = False
if Statistics:
    import scipy.stats as st
    import numpy as np
    import matplotlib.pyplot as plt

    test_size = 0.2

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=1234)

    # Baseline model
    yhatA = np.array([y_train.mean() for _ in range(y_test.shape[0])])

    # Linear regression with regularization term model
    yhatB = LinearRegressionFit(X_train, X_test, y_train, opt_lambda)

    # ANN
    yhatC = ANNFit(X_train, X_test, y_train, y_test, h)

    # perform statistical comparison of the models
    # compute z with squared error.
    zA = np.abs(y_test - yhatA)**2
    zB = np.abs(y_test - yhatB)**2
    zC = np.abs(y_test - yhatC)**2

    # compute confidence interval of model A
    alpha = 0.05
    # CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    zAB = zA - zB
    CIA = st.t.interval(1-alpha, len(zAB)-1, loc=np.mean(zAB),
                        scale=st.sem(zAB))  # Confidence interval
    p = 2*st.t.cdf(-np.abs(np.mean(zAB))/st.sem(zAB), df=len(zAB)-1)  # p-value
    print(
        "p-value for Baseline & Regularized Linear Regression = {0}".format(p))
    print("CI for Baseline & Regularized Linear Regression = {0}".format(CIA))

    zAC = zA - zC
    CIB = st.t.interval(1-alpha, len(zAC)-1, loc=np.mean(zAC),
                        scale=st.sem(zAC))  # Confidence interval
    p = 2*st.t.cdf(-np.abs(np.mean(zAC))/st.sem(zAC), df=len(zAC)-1)  # p-value
    print("p-value for Baseline & ANN  = {0}".format(p))
    print("CI for Baseline & ANN  = {0}".format(CIB))

    zBC = zB - zC
    CIC = st.t.interval(1-alpha, len(zBC)-1, loc=np.mean(zBC),
                        scale=st.sem(zBC))  # Confidence interval
    p = 2*st.t.cdf(-np.abs(np.mean(zBC))/st.sem(zBC), df=len(zBC)-1)  # p-value
    print("p-value for Regularized Linear Regression vs ANN= {0}".format(p))
    print("CI for Regularized Linear Regression vs ANN= {0}".format(CIC))

    data_dict = {}
    data_dict['category'] = ['Baseline vs Linear Regression',
                             'Baseline vs ANN', 'Linear Regression vs ANN']
    data_dict['lower'] = [CIA[0], CIB[0], CIC[0]]
    data_dict['upper'] = [CIA[1], CIB[1], CIC[1]]
    dataset = pd.DataFrame(data_dict)
    for lower, upper, y in zip(dataset['lower'], dataset['upper'], range(len(dataset))):
        plt.plot((y, y), (lower, upper), 'o-')
    plt.xticks(range(len(dataset)), [1, 2, 3])
    plt.legend(['Baseline vs Linear Regression',
               'Baseline vs ANN', 'Linear Regression vs ANN'], loc='upper right')
    plt.plot((-0.5, 3.5), (0, 0), 'k-')
    plt.xlim(-0.5, len(dataset)-0.5)
    plt.ylim(-0.3, 1.8)
    plt.xlabel("Models")
    plt.ylabel("Squared Error Difference")

    plt.show()
