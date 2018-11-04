import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers_abalone import   pairwiseDistCorr,nn_reg,nn_arch,reconstructionError
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product
import abalone_EMTestCluster as emtc
import abalone_KMeansTestCluster as kmtc
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.ticker import MaxNLocator


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
    abalone_data = pd.read_csv("abalone_data.csv")
    dft, mapping = encode_target(abalone_data, "rings")
    dft.to_csv('abalone_datanew.cvs')
    X = (dft.ix[:,:-1])
    y = dft.ix[:, -1]
    abalone_data.describe()
    
    #randomized projection
    tmp = defaultdict(dict)
    dims = [1,2,3,4,5,6,7,8]
    for i, dim in product(range(20), dims):
        rp = SparseRandomProjection(random_state=i, n_components=dim)
        tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(X), X)
    tmp = pd.DataFrame(tmp).T
    tmp
    tmp.to_csv('rp_abalone_iterations.csv')
    
    
    tmp_fit = defaultdict(dict)
    for i,dim in product(range(20),dims):
        rp = SparseRandomProjection(random_state=i, n_components=dim)
        rp.fit(X)  
        tmp_fit[dim][i] = reconstructionError(rp, X)
    tmp_fit =pd.DataFrame(tmp_fit).T
    tmp_fit
    tmp_fit.to_csv('rp_abalone_new_data.csv')
	grid ={'rp__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
	rp = SparseRandomProjection(random_state=10)       
	mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
	pipe = Pipeline([('rp',rp),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(X,y)
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv('rp_abalone_ann.csv')
	#ndim= 3
	dim = 3
	rp = SparseRandomProjection(n_components=dim,random_state=10)
	X2 = rp.fit_transform(X)
	rp_abalone_new_data = pd.DataFrame(np.hstack((X2,np.atleast_2d(y).T)))
	cols = list(range(rp_abalone_new_data.shape[1]))
	cols[-1] = 'rings'
	rp_abalone_new_data.columns = cols
	rp_abalone_new_data.to_csv('rp_abalone_new_data.csv')
	###KMeans###
	tester = kmtc.KMeansTestCluster(X2, y, clusters=range(1,41), plot=True, targetcluster=5, stats=True)
	tester.run()

	###EM####
	tester = emtc.ExpectationMaximizationTestCluster(X2, y, clusters=range(1,41), plot=True, targetcluster=5, stats=True)
	tester.run()
	nclust = 5
	k_means = KMeans(nclust)
	k_means.fit(X2)
	clust_labels = k_means.predict(X2)
	cent = k_means.cluster_centers_
	kmeans = pd.DataFrame(clust_labels)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	scatter = ax.scatter(X2[:,0], X2[:,1], c = kmeans[0], s = 10, cmap='viridis', marker='*')
	plt.scatter(cent[:, 0], cent[:, 1], c='red', s=200, alpha=0.5)
	ax.set_title('Abalone: K-Means Clustering (after RP)')
	ax.set_xlabel('RP1')
	ax.set_ylabel('RP2')
	plt.colorbar(scatter)


	#EM
	em = GMM(covariance_type = 'diag')
	em.set_params(n_components = nclust)
	em.fit(X2)
	labels =  pd.DataFrame(em.predict(X2))

	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	scatter2 = ax2.scatter(X2[:,0], X2[:,1], c = labels[0], s = 10, cmap='viridis')
	#plt.scatter(cent_em[:, 0], cent_em[:, 1], c='black', s=200, alpha=0.5)
	ax2.set_title('Abalone: EM Clustering (after RP)')
	ax2.set_xlabel('RP1')
	ax2.set_ylabel('RP2')
	plt.colorbar(scatter2)
