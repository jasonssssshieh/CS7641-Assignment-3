import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, decomposition
from sklearn.decomposition import FastICA
import mushroom_EMTestCluster as emtc
import mushroom_KMeansTestCluster as kmtc
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers_abalone import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA

def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column].replace(map_to_int, inplace=True)
    return (df_mod, map_to_int)

if __name__ == "__main__":
    mushroom_data = pd.read_csv("mushroom_data.csv")
    dft, mapping = encode_target(mushroom_data, "class")
    dft.to_csv('mushroom_datanew.cvs')
    X = (dft.ix[:,:-1])
    y = dft.ix[:, -1]
    mushroom_data.describe()
    
    
    ica = FastICA(random_state = 5)
    kurt = {}
    clusters = range(1,41)
    dims = range(1,23)
    
    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt[dim] = tmp.abs().mean()
    kurt = -np.sort(-pd.Series(kurt))
    print(kurt)
    fig, ax = plt.subplots()
    ax.bar(range(1, 1+len(kurt)), kurt)
    ax.set_ylabel('Kurtosis')
    ax.set_title('Mushroom: ICA Kurtosis')
    plt.show()
	
	grid ={'ica__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
	ica = FastICA(random_state=5)       
	mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
	pipe = Pipeline([('ica',ica),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
	gs.fit(X,y)
	tmp = pd.DataFrame(gs.cv_results_)
    ica_mushroom_ann.to_csv('ica_mushroom_ann.csv')

	dim = 12
	ica = FastICA(n_components=dim,random_state=5)
	X2 = ica.fit_transform(X)
	ica_mushroom_new_data = pd.DataFrame(np.hstack((X2,np.atleast_2d(y).T)))
	cols = list(range(ica_mushroom_new_data.shape[1]))
	cols[-1] = 'class'
	ica_mushroom_new_data.columns = cols
	ica_mushroom_new_data.to_csv('ica_mushroom_new_data.csv')
	
	    ###KMeans###
    tester = kmtc.KMeansTestCluster(X2, y, clusters=range(1,35), plot=True, targetcluster=2, stats=True)
    tester.run()

###EM####
    tester = emtc.ExpectationMaximizationTestCluster(X2, y, clusters=range(1,35), plot=True, targetcluster=2, stats=True)
    tester.run()
    
	#plot the scatters
	#kmeans
    nclust = 12
    k_means = KMeans(nclust)
    k_means.fit(X2)
    clust_labels = k_means.predict(X2)
    cent = k_means.cluster_centers_
    kmeans = pd.DataFrame(clust_labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X2[:,1], X2[:,2], c = kmeans[0], s = 10, cmap='viridis', marker='p')
    plt.scatter(cent[:, 0], cent[:, 1], c='yellow', s=200, alpha=0.5)
    ax.set_title('Mushroom: K-Means Clustering (after ICA)')
    ax.set_xlabel('IC1')
    ax.set_ylabel('IC2')
    plt.colorbar(scatter)

	#EM
    em = GMM(covariance_type = 'diag')
    em.set_params(n_components = nclust)
    em.fit(X2)
    labels =  pd.DataFrame(em.predict(X2))
    
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	scatter2 = ax2.scatter(X2[:,1], X2[:,2], c = labels[0], s = 10, cmap='viridis')
	#plt.scatter(cent_em[:, 0], cent_em[:, 1], c='black', s=200, alpha=0.5)
	ax2.set_title('Mushroom: EM Clustering (after ICA)')
	ax2.set_xlabel('IC1')
	ax2.set_ylabel('IC2')
	plt.colorbar(scatter2)
