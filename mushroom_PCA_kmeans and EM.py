import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers_mushroom import  nn_arch,nn_reg
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import mushroom_EMTestCluster as emtc
import mushroom_KMeansTestCluster as kmtc
import matplotlib.pyplot as plt
from sklearn import metrics, decomposition
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler

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
    dft.to_csv('mushroom_newdata.cvs')
    X = (dft.ix[:,:-1])
    X = StandardScaler().fit_transform(X)
    y = dft.ix[:, -1]
    clusters =  [2,5,10,15,20,25,30,35,40]
    dims = range(1,22)
    pca = PCA(random_state=5)
    pca.fit(X)
    tmp = pd.Series(data = pca.explained_variance_,index = range(0,22))
    tmp.to_csv('PCA_mushroom_X.csv')
    print(pca.explained_variance_ratio_)
    fig, ax = plt.subplots()
    ax.bar(range(1, 1+len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    ax.set_ylabel('pca: explained_variance_')
    ax.set_title('Mushroom: PCA explained_variance_ratio_')
    plt.show()
    
    pca_explained_variance_ratio_cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots()
    ax.bar(range(1, 1+len(pca_explained_variance_ratio_cum)), pca_explained_variance_ratio_cum)
    ax.set_ylabel('pca: explained_variance_ratio_cumulative')
    ax.set_title('Mushroom: PCA explained_variance_ratio_cumulative')
    plt.show()

    grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
	pca = PCA(random_state=5)       
	mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
	pipe = Pipeline([('pca',pca),('NN',mlp)])

	grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
	pca = PCA(random_state=5)       
	mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
	pipe = Pipeline([('pca',pca),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
	gs.fit(X,y)
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv('pca_mushroom_ann.csv')

	#n_dim = 9 is the best for prediction

	dim = 9
	pca = PCA(n_components=dim,random_state=5)
	X2 = pca.fit_transform(X)
	mushroom_data_2 = pd.DataFrame(np.hstack((X2,np.atleast_2d(y).T)))
	cols = list(range(mushroom_data_2.shape[1]))
	cols[-1] = 'class'
	mushroom_data_2.columns = cols
	mushroom_data_2.to_csv('pca_mushroom_new_data.csv')
	
	###KMeans###
	tester = kmtc.KMeansTestCluster(X2, y, clusters=range(1,35), plot=True, targetcluster=2, stats=True)
	tester.run()
	###EM####
	tester = emtc.ExpectationMaximizationTestCluster(X2, y, clusters=range(1,35), plot=True, targetcluster=2, stats=True)
	tester.run()

	#plot the scatters
	#kmeans
	nclust = 8
	k_means = KMeans(nclust)
	k_means.fit(X2)
	clust_labels = k_means.predict(X2)
	cent = k_means.cluster_centers_
	kmeans = pd.DataFrame(clust_labels)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	scatter = ax.scatter(X2[:,0], X2[:,1], c = kmeans[0], s = 10, cmap='viridis')
	plt.scatter(cent[:, 0], cent[:, 1], c='black', s=200, alpha=0.5)
	ax.set_title('Mushroom: K-Means Clustering (after PCA)')
	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')
	#EM
	em = GMM(covariance_type = 'diag')
	em.set_params(n_components = nclust)
	em.fit(X2)
	labels =  pd.DataFrame(em.predict(X2))

	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	scatter2 = ax2.scatter(X2[:,0], X2[:,1], c = labels[0], s = 10, cmap='viridis')
	#plt.scatter(cent_em[:, 0], cent_em[:, 1], c='black', s=200, alpha=0.5)
	ax2.set_title('Mushroom: EM Clustering (after PCA)')
	ax2.set_xlabel('PC1')
	ax2.set_ylabel('PC2')
	plt.colorbar(scatter2)