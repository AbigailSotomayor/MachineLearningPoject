import pandas as pd
from methods import *
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
import torch
from sklearn import model_selection 


#Reading data
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

K_Outter = 5
K_Inner = 5
hs = [1,2,3,4,5]
CV_Outter = model_selection.KFold(K_Outter, shuffle=True,random_state=1234)

Error_test_baseline = [0 for _ in range(K_Outter)]
Error_test_LR = [0 for _ in range(K_Outter)]
best_lambda = [0 for _ in range(K_Outter)]
best_hs = [0 for _ in range(K_Outter)]
Error_test_ANN = [0 for _ in range(K_Outter)]

for k, (train_index, test_index) in enumerate(CV_Outter.split(X,y)):

    print("Outter cross validation {0}/{1}".format(k+1,K_Outter))

    for i in range(3):
        if i == 0:
            print("Baseline")
            Error_test_baseline[k] = BaselineCrossValidation(train_index,test_index,y)
            print("Inner cross validation done.\n")
        elif i == 1:
            print("Linear Regression")
            lambdas = np.power(10.,range(-4,8))
            Error_test_LR[k], best_lambda[k] = LinearRegressionCrossValidation(train_index, test_index, X, y, K_Inner,lambdas)
            print("Inner cross validation done.\n")
            
        elif i == 2:
            print("ANN")
            best_hs[k], Error_test_ANN[k]  = ANNCrossValidation(train_index, test_index, X, y, hs, K_Inner, K_Outter)
            print("Inner cross validation done.\n")


print("Summary:")

print("==================================================================")
print("Baseline:")
for index,val in enumerate(Error_test_baseline):
    print("Fold {0}: {1}".format(index+1,val))
print("Mean: " + str(np.mean(Error_test_baseline)))
print("==================================================================\n")

print("==================================================================")
print("Linear Regression")
for index,val in enumerate(Error_test_LR):
        print("Fold {0}: Error {1}".format(index+1,val))
print("Mean: " + str(np.mean(Error_test_LR)))
print("Best Lambdas")
for index,val in enumerate(best_lambda):
        print("Fold {0}: Error {1}".format(index+1,val))
print("Mean: " + str(np.mean(best_lambda)))
print("==================================================================\n")

print("==================================================================")
print("Hidden units with corresponding MSE are: ")
for i in range(len(best_hs)):
    print("Best h in the outter fold {0} = {1} with the generalization error E_gen={2}".format(i+1,best_hs[i],Error_test_ANN[i][0]))
print("Mean Error: {0}".format(np.mean(Error_test_ANN)))
print("==================================================================")


h = best_hs[np.argmin(Error_test_ANN)]
opt_lambda = best_lambda[np.argmin(Error_test_LR)]

Statistics = True
if Statistics:
    import scipy.stats as st 
    import numpy as np
    import matplotlib.pyplot as plt

    test_proportion = 0.2

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion,random_state=1234)
    
    # Baseline model
    yhatA = np.array([y_train.mean() for _ in range(y_test.shape[0])])
    
    # Linear regression with regularization term model
    yhatB = LinearRegressionFit(X_train, X_test, y_train, opt_lambda)
    
    #ANN 
    yhatC = ANNFit(X_train, X_test, y_train, y_test, h)

    # perform statistical comparison of the models
    # compute z with squared error.
    zA = np.abs(y_test - yhatA ) ** 2
    zB = np.abs(y_test - yhatB) ** 2
    zC = np.abs(y_test - yhatC) ** 2

    # compute confidence interval of model A
    alpha = 0.05
    # CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    zAB = zA - zB
    CI = st.t.interval(1-alpha, len(zAB)-1, loc=np.mean(zAB), scale=st.sem(zAB))  # Confidence interval
    p = 2*st.t.cdf( -np.abs( np.mean(zAB) )/st.sem(zAB), df=len(zAB)-1)  # p-value
    print("p-value for Baseline & Regularized Linear Regression = {0}".format(p))
    print("CI for Baseline & Regularized Linear Regression = {0}".format(CI))
    
    zAC = zA - zC
    CI = st.t.interval(1-alpha, len(zAC)-1, loc=np.mean(zAC), scale=st.sem(zAC))  # Confidence interval
    p = 2*st.t.cdf( -np.abs( np.mean(zAC) )/st.sem(zAC), df=len(zAC)-1)  # p-value
    print("p-value for Baseline & ANN  = {0}".format(p))
    print("CI for Baseline & ANN  = {0}".format(CI))
    
    zBC = zB - zC
    CI = st.t.interval(1-alpha, len(zBC)-1, loc=np.mean(zBC), scale=st.sem(zBC))  # Confidence interval
    p = 2*st.t.cdf( -np.abs( np.mean(zBC) )/st.sem(zBC), df=len(zBC)-1)  # p-value
    print("p-value for Regularized Linear Regression vs ANN= {0}".format(p))
    print("CI for Regularized Linear Regression vs ANN= {0}".format(CI))



    





# Summary:
# ==================================================================
# Baseline:
# Fold 1: 2.7008794542721706
# Fold 2: 5.259218268007862
# Fold 3: 4.800367993856332
# Fold 4: 4.724234593572779
# Fold 5: 3.3557857632325145
# Mean: 4.1680972145883315
# ==================================================================

# ==================================================================
# Linear Regression
# Fold 1: Error 2.205621833068359
# Fold 2: Error 4.490813039063822
# Fold 3: Error 4.210516465169038
# Fold 4: Error 3.976710907811849
# Fold 5: Error 2.5698909467267925
# Mean: 3.4907106383679727
# Best Lambdas
# Fold 1: Error 10.0
# Fold 2: Error 10.0
# Fold 3: Error 10.0
# Fold 4: Error 100.0
# Fold 5: Error 100.0
# Mean: 46.0
# ==================================================================

# ==================================================================
# Hidden units with corresponding MSE are:
# Best h in the outter fold 1=8 with the generalization error E_gen=[2.6653972]
# Best h in the outter fold 2=9 with the generalization error E_gen=[5.365884]
# Best h in the outter fold 3=10 with the generalization error E_gen=[4.9660673]
# Best h in the outter fold 4=6 with the generalization error E_gen=[4.713299]                                                        
# Best h in the outter fold 5=7 with the generalization error E_gen=[2.495098]
# Mean: 4.0411491
# ==================================================================

# p-value for Baseline & Regularized Linear Regression = 0.01319175117786182
# p-value for Regularized Linear Regression & ANN  = 0.5887362535265308
# p-value for Regularized Linear Regression vs ANN= 0.017238636761557232



# 1 test
# p-value for Baseline & Regularized Linear Regression = 5.0011157170177e-05
# p-value for Baseline & ANN  = 0.8223143259214822
# p-value for Regularized Linear Regression vs ANN= 2.528332160144012e-05

# 2 test
# p-value for Baseline & Regularized Linear Regression = 0.0804778453996975
# p-value for Baseline & ANN  = 0.735731993630482
# p-value for Regularized Linear Regression vs ANN= 0.08102105411644706

# 3 test with confidence interval. 
# p-value for Baseline & Regularized Linear Regression = 0.008283032582001346. Reject
# CI for Baseline & Regularized Linear Regression = (0.26564358554524203, 1.7464037434374817)
# p-value for Baseline & ANN  = 0.07152452850032992. Accept
# CI for Baseline & ANN  = (-0.04075654049510313, 0.0017430419942319024)
# p-value for Regularized Linear Regression vs ANN= 0.008291283415787947. Reject
# CI for Regularized Linear Regression vs ANN= (-1.7803666025833187, -0.27069422490027606)