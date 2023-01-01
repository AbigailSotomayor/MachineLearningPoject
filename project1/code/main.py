
from cProfile import label
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns


def basicStatistics(data):
    """Return the mean, median, and standard deviation of the data."""
    mean = np.mean(data, axis=0)
    median = np.median(data, axis=0)
    std = np.std(data, axis=0)
    return mean, median, std


def normalizeData(data):
    """Normalize the data to zero mean."""
    mean, _, std = basicStatistics(data)
    return (data - mean)/std


def svd(data):
    """Perform singular value decomposition on the data."""
    Y = data - np.ones((len(data), 1)) * np.mean(data,
                                                 axis=0)
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    V = Vh.T  # transpose
    return U, S, V


def varianceExplained(S):
    """Return the variance explained by the principal components."""
    return S**2 / ((S**2).sum())


def plotVarianceExplained(rho):
    """Plot the variance explained by the principal components."""
    print(np.cumsum(rho))
    threshold = 0.9
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(rho)+1), rho, 'x-')
    plt.plot(range(1, len(rho)+1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()


def projectData(data, V):
    """Project the centered data onto the principal components."""
    return data @ V


def plotPCA(m, n, data, classNames):
    """Plot the principal components."""
    import matplotlib.pyplot as plt
    _, _, V = svd(data)
    Z = projectData(data, V)
    C = len(classNames)
    plt.figure()
    for c in range(C):
        class_mask = data[:, -1] == c  # mask for chd = 0 or 1
        plt.plot(Z[class_mask, m-1], Z[class_mask, n-1], 'x')
    plt.legend(['sbp >= 160', 'sbp < 160'])
    plt.xlabel('PC{0}'.format(n))
    plt.ylabel('PC{0}'.format(m))
    plt.show()


def boxplot(data, attributes):
    """Plot boxplots of the data."""
    import matplotlib.pyplot as plt
    plt.figure()
    plt.boxplot(data, labels=attributes)
    plt.show()


def histogram(data, attributes):
    """Plot histograms of the data."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 5, figsize=(12, 12))
    for i in range(len(attributes)):
        sns.histplot(data, x=attributes[i], ax=axes[i//5, i % 5], kde=True)
    # vertical spacing
    fig.tight_layout(h_pad=2)
    plt.show()


def pairPlot(data):
    """Plot pairwise scatter plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    # exclude chd attribute
    data = excludeBinary(data)

    sns.set(style='ticks', color_codes=True)
    sns.pairplot(
        data, vars=data.columns[data.columns != 'chd'], corner=True, diag_kind='kde')
    plt.show()


def correlationMatrix(data):
    """Plot correlation matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    data = excludeBinary(data)
    sns.set(style='ticks', color_codes=True)
    sns.heatmap(data.corr(), annot=True)
    plt.show()


def plotThreeAttributes(data, attr1, attr2, attr3):
    """Plot three attributes against each other."""
    # plot3d
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[attr1], data[attr2], data[attr3], c=data['chd'])
    ax.set_xlabel(attr1)
    ax.set_ylabel(attr2)
    ax.set_zlabel(attr3)
    plt.show()


def plotThreePCAs(data, i, j, k):
    """Plot three principal components against each other."""
    import matplotlib.pyplot as plt
    _, _, V = svd(np.array(data))
    Z = projectData(np.array(data), V)
    ax = plt.axes(projection='3d')
    ax.scatter3D(Z[data['chd'] == 0, i-1], Z[data['chd'] == 0,
                 j-1], Z[data['chd'] == 0, k-1], label='No CHD', marker='x')
    ax.scatter3D(Z[data['chd'] == 1, i-1], Z[data['chd'] == 1,
                 j-1], Z[data['chd'] == 1, k-1], label='CHD', marker='x')
    ax.set_xlabel('PC{0}'.format(i))
    ax.set_ylabel('PC{0}'.format(j))
    ax.set_zlabel('PC{0}'.format(k))
    ax.legend()
    plt.show()


# plotting coefficients of PCA
def plotCoefficients(V, attributes):
    import matplotlib.pyplot as plt

    pcs = [0, 1, 2]
    legendStrs = ['PC'+str(e+1) for e in pcs]
    bw = .2
    r = np.arange(1, len(attributes)+1)
    for i in pcs:
        plt.bar(r+i*bw, V[:, i], width=bw)
    plt.xticks(r+bw, attributes)
    plt.xlabel('Attributes')
    plt.ylabel('Component coefficients')
    plt.legend(legendStrs)
    plt.title('PCA Component Coefficients')
    plt.show()


def convertData(data):
    """Convert text to numbers."""
    data.drop('row.names', axis=1, inplace=True)
    data['famhist'] = data['famhist'].map({'Present': 1, 'Absent': 0})
    return data


def getAttributes(data):
    return np.array(data.columns)


def excludeBinary(data):
    """Exclude binary attributes."""
    return data.drop(['famhist'], axis=1)


def qqPlot(data, attribute):
    """Plot a qq-plot of a single attribute."""
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    data = np.array(data[attribute])
    plt.figure()
    stats.probplot(data, dist='norm', plot=plt)
    plt.show()


data = pd.read_csv('data.csv')
data = convertData(data)
data.describe().to_clipboard(index=False)
attributes = getAttributes(data)
classNames = np.unique(data['chd'])

# boxplot(data, attributes)
# sns.histplot(data, common_bins=False, kde=True, multiple='dodge')
# sns.boxplot(data)

# histogram(data, attributes)
# pairPlot(data)
# plotPCA(1, 2, np.array(data), classNames)
# plotVarianceExplained(varianceExplained(svd(np.array(data))[1]))
# plotThreeAttributes(data, 'alcohol', 'age', 'sbp')
# plotThreePCAs(data, 1, 2, 3)
# correlationMatrix(data)

# plotCoefficients(svd(np.array(data))[2], attributes)


# print(svd(np.array(data))[2])
# qqPlot(data, 'sbp')
# qqPlot(data, 'typea')
