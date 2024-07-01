from collections import defaultdict, OrderedDict, Counter
import numpy as np
from scipy import optimize
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
# from sklearn.cluster import DBSCAN
from cuml.cluster import DBSCAN
from sklearn.base import BaseEstimator, ClusterMixin
from joblib import Parallel, delayed

from sklearn.metrics.pairwise import pairwise_distances
from itertools import combinations
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
import json

def scalable_silhouette_score(X, labels, metric='euclidean', sample_size=None,
                              random_state=None, n_jobs=1, **kwds):
    if sample_size is not None:
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            raise ValueError('Distance matrix cannot be precomputed')
        else:
            X, labels = X[indices], labels[indices]

    return np.mean(scalable_silhouette_samples(
        X, labels, metric=metric, n_jobs=n_jobs, **kwds))

def scalable_silhouette_samples(X, labels, metric='euclidean', n_jobs=1, **kwds):
    A = _intra_cluster_distances_block(X, labels, metric, n_jobs=n_jobs, **kwds)
    B = _nearest_cluster_distance_block(X, labels, metric, n_jobs=n_jobs, **kwds)
    sil_samples = (B - A) / np.maximum(A, B)
    return np.nan_to_num(sil_samples)

def _intra_cluster_distances_block(X, labels, metric, n_jobs=1, **kwds):
    intra_dist = np.zeros(labels.size, dtype=float)
    values = Parallel(n_jobs=n_jobs)(
            delayed(_intra_cluster_distances_block_)
                (X[np.where(labels == label)[0]], metric, **kwds)
                for label in np.unique(labels))
    for label, values_ in zip(np.unique(labels), values):
        intra_dist[np.where(labels == label)[0]] = values_
    return intra_dist

def _nearest_cluster_distance_block(X, labels, metric, n_jobs=1, **kwds):
    inter_dist = np.empty(labels.size, dtype=float)
    inter_dist.fill(np.inf)
    unique_labels = np.unique(labels)

    values = Parallel(n_jobs=n_jobs)(
        delayed(_nearest_cluster_distance_block_)(
            X[np.where(labels == label_a)[0]],
            X[np.where(labels == label_b)[0]],
            metric, **kwds)
        for label_a, label_b in combinations(unique_labels, 2))

    for (label_a, label_b), (values_a, values_b) in zip(combinations(unique_labels, 2), values):
        indices_a = np.where(labels == label_a)[0]
        inter_dist[indices_a] = np.minimum(values_a, inter_dist[indices_a])
        del indices_a
        indices_b = np.where(labels == label_b)[0]
        inter_dist[indices_b] = np.minimum(values_b, inter_dist[indices_b])
        del indices_b
    return inter_dist

def _intra_cluster_distances_block_(subX, metric, **kwds):
    distances = pairwise_distances(subX, metric=metric, **kwds)
    return distances.sum(axis=1) / (distances.shape[0] - 1)

def _nearest_cluster_distance_block_(subX_a, subX_b, metric, **kwds):
    dist = pairwise_distances(subX_a, subX_b, metric=metric, **kwds)
    dist_a = dist.mean(axis=1)
    dist_b = dist.mean(axis=0)
    return dist_a, dist_b

class Clusterer(BaseEstimator, ClusterMixin):
    """Performance clustering

    Parameters
    ----------
    eps_range: tuple (pair)
        the minimum and the maximum `eps` to try when choosing the best value of `eps`
        (the one having the best silhouette score)

    min_samples: int
        The number of samples in a neighborhood for a point to be considered as a core point.

    border_threshold: float
        the threshold to use for selecting the borderline.
        It indicates the max silhouette for a borderline point.

    verbose: boolean
        verbosity mode.
        default: False

    random_state : int
        RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    sample_size : int
        the number of samples (rows) that must be used when computing the silhouette score (the function silhouette_score is computationally expensive and generates a Memory Error when the number of samples is too high)
        default: 10000

    max_rows : int
        the maximum number of samples (rows) to be considered for the clustering task (the function silhouette_samples is computationally expensive and generates a Memory Error when the input matrix have too many rows)
        default: 40000


    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    n_clusters_: int
        number of clusters found by the algorithm
    labels_ :
        Labels of each point
    eps_range: tuple
        minimum and maximum `eps` to try
    min_samples: int
        the number of samples in a neighborhood for a point to be considered as a core point.
    verbose: boolean
        whether or not to show details of the execution
    random_state: int
        RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by 'np.random'.
    sample_size: None
    dbscan: scikit-learn DBSCAN object
    """

    def __init__(self, eps_range=(0.1, 5.0), min_samples=5, border_threshold=0.2, verbose=False, random_state=42,
                 sample_size=None):
        self.eps_range = eps_range
        self.min_samples = min_samples
        self.border_threshold = border_threshold
        self.verbose = verbose
        self.sample_size = sample_size
        self.random_state = random_state
        self.labels_ = []

    def _find_clusters(self, X, make_plot=True):
        if self.verbose:
            print('FITTING DBSCAN...\n')
            print('eps\t|silhouette')
            print('---------------------')

        self.eps2silhouettes_ = {}
        eps_min, eps_max = self.eps_range
        best_eps, best_silhouette = 0, 0.0
        for eps in np.linspace(eps_min, eps_max, num=20):

            # computation
            dbscan = DBSCAN(eps=eps, min_samples=self.min_samples, n_jobs=-1)
            dbscan.fit(X)
            cluster_labels = dbscan.labels_

            silhouette = scalable_silhouette_score(X, cluster_labels,
                                                   sample_size=self.sample_size,
                                                   random_state=self.random_state)
            if self.verbose:
                print(f'{eps:.2f}\t|{round(silhouette, 4)}')

            if silhouette >= best_silhouette:
                best_silhouette = silhouette
                best_eps = eps

            self.eps2silhouettes_[eps] = silhouette

        self.dbscan_ = DBSCAN(eps=best_eps, min_samples=self.min_samples, n_jobs=-1)
        self.dbscan_.fit(X)
        self.labels_ = self.dbscan_.labels_
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)

        if self.verbose:
            print(f'Best: eps={best_eps} (silhouette={round(best_silhouette, 4)})\n')

    def _cluster_borderline(self, X):
        """
        Assign clusters to borderline points, according to the borderline_threshold
        specified in the constructor.
        """
        if self.verbose:
            print('FINDING hybrid centers of performance...\n')

        self.labels_ = [[] for _ in range(len(X))]

        ss = scalable_silhouette_samples(X, self.dbscan_.labels_)
        for i, (row, silhouette, cluster_label) in enumerate(zip(X, ss, self.dbscan_.labels_)):
            if silhouette >= self.border_threshold:
                self.labels_[i].append(cluster_label)
            else:
                intra_distances = []
                inter_distances = []
                for label in set(self.dbscan_.labels_):
                    if label == -1:
                        continue  # skip noise points
                    cluster_points = X[self.dbscan_.labels_ == label]
                    intra_distances.append(np.mean([euclidean(row, point) for point in cluster_points]))

                for label in set(self.dbscan_.labels_):
                    if label == -1 or label == cluster_label:
                        continue  # skip noise points and the same cluster
                    cluster_points = X[self.dbscan_.labels_ == label]
                    inter_distances.append(np.mean([euclidean(row, point) for point in cluster_points]))

                intra_silhouette = np.min(intra_distances) if intra_distances else float('inf')
                inter_silhouette = np.min(inter_distances) if inter_distances else float('inf')
                
                silhouette = (inter_silhouette - intra_silhouette) / max(inter_silhouette, intra_silhouette)
                if silhouette <= self.border_threshold:
                    nearest_cluster_label = np.argmin(inter_distances)
                    self.labels_[i].append(nearest_cluster_label)

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
            for row, labels in zip(X, self.kmeans_.predict(X)):
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
        """
        Compute performance clustering.

        Parameters
        ----------
            X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.

            kind: str
                single: single cluster
                multi: multi cluster

            y: ignored
        """
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
        """
        Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        multi_labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        if self.kind_ == 'single':
            return self.kmeans_predict(X)
        else:
            multi_labels = []
            for row in X:
                x, y = tuple(row)
                labels = self._matrix[(int(x), int(y))]
                multi_labels.append(labels)
            return multi_labels


