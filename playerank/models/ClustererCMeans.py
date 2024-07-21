from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import mutual_info_score, adjusted_rand_score

def calculate_silhouette_score(data, cluster_membership):
    # Calculate and print the average silhouette score for each cluster
    silhouette_avg = silhouette_score(data, cluster_membership)
    davies_bouldin_metric = davies_bouldin_score(data, cluster_membership)
    calinski_harabasz_metric = calinski_harabasz_score(data, cluster_membership)
    mutual_info_metric = mutual_info_score(data, cluster_membership)
    adjusted_rand_metric = adjusted_rand_score(data, cluster_membership)

    print(f"davies_bouldin_score: {davies_bouldin_metric}")
    print(f"calinski_harabasz_score: {calinski_harabasz_metric}")
    print(f"mutual_info_score: {mutual_info_metric}")
    print(f"adjuster_rand_score: {adjusted_rand_metric}")
    return silhouette_avg

class Clusterer(BaseEstimator, ClusterMixin):
    
    def __init__(self, k_range=(2, 15), border_threshold=0.1, verbose=False, random_state=42,
                sample_size=None):
        self.k_range = k_range
        self.border_threshold = border_threshold
        self.verbose = verbose
        self.sample_size = sample_size
        # initialize attributes
        self.labels_ = []
        self.random_state = random_state

    def _find_clusters(self, X, make_plot=True):
        if self.verbose:
            print ('FITTING kmeans...\n')
            print ('n_clust\t|silhouette')
            print ('---------------------')

        self.k2silhouettes_ = {}
        kmin, kmax = self.k_range
        range_n_clusters = range(kmin, kmax + 1)
        best_k, best_silhouette = 0, 0.0
        for k in range_n_clusters:

            # computation
            center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                X.T,
                k,
                2,
                error=0.005,
                maxiter=100000, 
                init=None
            )
            
            cluster_membership = np.argmax(u, axis=0)

            silhouette = calculate_silhouette_score(X, cluster_membership)

            if self.verbose:
                print ('%s\t|%s' % (k, round(silhouette, 4)))

            if silhouette >= best_silhouette:
                best_silhouette = silhouette
                best_k = k
                #best_silhouette_samples = ss

            self.k2silhouettes_[k] = silhouette


        best_center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                X.T,
                best_k,
                2,
                error=0.005,
                maxiter=100000, 
                init=None
            )        
        self.n_clusters_ = best_k
        self.cluster_centers_ = best_center

        if self.verbose:
            print ('Best: n_clust=%s (silhouette=%s)\n' % (best_k, round(best_silhouette, 4)))

    def _cluster_borderline(self, X, cluster_membership):
        """
        Assign clusters to borderline points, according to the borderline_threshold
        specified in the constructor
        """
        if self.verbose:
            print ('FINDING hybrid centers of performance...\n')

        self.labels_ = [[] for i in range(len(X))]

        ss = silhouette_samples(X, cluster_membership)

        for i, (row, silhouette, cluster_label) in enumerate(zip(X, ss, self.kmeans_.labels_)):
            if silhouette >= self.border_threshold:
                self.labels_[i].append(cluster_label)
            else:
                intra_silhouette = euclidean(row, self.kmeans_.cluster_centers_[cluster_label])
                for label in set(self.kmeans_.labels_):
                    inter_silhouette = euclidean(row, self.kmeans_.cluster_centers_[label])
                    silhouette = (inter_silhouette - intra_silhouette) / max(inter_silhouette, intra_silhouette)
                    if silhouette <= self.border_threshold:
                        self.labels_[i].append(label)

        return ss

    def _generate_matrix(self, ss, kind = 'multi'):
        """
        Generate a matrix for optimizing the predict function
        """
        matrix = {}
        X = []

        for i in range(0, 101):
            for j in range(0, 101):
                X.append([i, j])
        if kind == 'multi':
            multi_labels = self._predict_with_silhouette(X, ss)
            for row, labels in zip(X, multi_labels):
                matrix[tuple(row)] = labels
        else:
            for row, labels in zip(X, self.predict(X)):
                matrix[tuple(row)] = labels
        self._matrix = matrix

    def get_clusters_matrix(self, kind = 'single'):
        roles_matrix = {}
        m= self._matrix.items()
        # if kind != 'single':
        #     m= self._matrix.items()
        #
        # else:
        #     m = self._matrix_single.items()

        for k,v in  m:
            x,y = int(k[0]),int(k[1])
            if k[0] not in roles_matrix:
                roles_matrix[x] = {}
            roles_matrix[x][y] = "-".join(map(str,v)) if kind !='single' else int(v) #casting with python int, otherwise it's not json serializable
        return roles_matrix

    def fit(self, player_ids, match_ids, dataframe, y=None, kind='single', filename='clusters'):
        
        self.kind_ = kind
        X = dataframe.values

        self._find_clusters(X)      # find the clusters with kmeans
        if kind != 'single':


            silhouette_scores = self._cluster_borderline(X) # assign multiclusters to borderline performances
            self._generate_matrix(silhouette_scores)    # generate the matrix for optimizing the predict function
        else:
            self._generate_matrix(None, kind = 'single') #no silhouette scores if kind single
        if self.verbose:
            print ("DONE.")




        return self

    def _predict_with_silhouette(self, X, ss):
        cluster_labels, threshold = self.kmeans_.predict(X), self.border_threshold
        multicluster_labels = [[] for _ in cluster_labels]
        if len(set(cluster_labels)) == 1:
            return [[cluster_label] for cluster_label in cluster_labels]
        for i, (row, silhouette, cluster_label) in enumerate(zip(X, ss, cluster_labels)):
            if silhouette >= threshold:
                multicluster_labels[i].append(cluster_label)
            else:
                intra_silhouette = euclidean(row, self.cluster_centers_[cluster_label])
                for label in set(cluster_labels):
                    inter_silhouette = euclidean(row, self.cluster_centers_[label])
                    silhouette = (inter_silhouette - intra_silhouette) / max(inter_silhouette, intra_silhouette)
                    if silhouette <= threshold:
                        multicluster_labels[i].append(label)

        return np.array(multicluster_labels)

    def predict(self, X, y=None):
        X = pd.DataFrame(X)
        X = X.values
        if self.kind_ == 'single':
            u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(X.T, self.cluster_centers_, 2, error=0.005, maxiter=1000)
            cluster_membership = np.argmax(u, axis=0)
            return cluster_membership
        
