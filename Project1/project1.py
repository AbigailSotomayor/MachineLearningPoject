import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv('https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data')

df = df.drop(["row.names"], axis = 1)

#since famhist is not an integer value we convert it
#so that 0:Absent and 1:Present
df["famhist"].replace("Absent", 0, inplace = True)
df["famhist"].replace("Present", 1, inplace = True)

#we would like to have the dataframe as a matrix

raw_data = df.values
cols = range(0,10)
X = raw_data[:, cols]

attributeNames = np.asarray(df.columns[cols])

#Now we would like to find out if there is any outliers
#for this reason we would like to plot a boxplot of the
#attributes in this dataframe 
M = len(attributeNames)
N = len(X[:,0])

# sbp = X[:,0]
# tobacco = X[:,1]
# ldl = X[:,2]
# adiposity = X[:,3]
# famhist = X[:,4]
# typea = X[:,5]
# besity = X[:,6]
# alcohol = X[:,7]
#age = X[:,8]
# chd = X[:,9]

plt.boxplot(X)
plt.xticks(range(1,M+1), attributeNames, rotation=45)
plt.show()

#for the outliers check we will not consider the attributes of
#famhist, chd, age and adiposity as this ones from the boxplot can be seen 
#to clearly do not have any outliers.
