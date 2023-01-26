import numpy as np

class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """
        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        while iteration < self.max_iter:
            # your code
            iteration = iteration + 1
            dist = self.euclidean_distance(X, self.centroids)
            clustering = np.argmin(dist, axis=1) # the cluster index is the one with the minimum distance
            self.update_centroids(clustering, X) # see helper function below
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        #your code
        k = self.centroids.shape[0]
        centroids = np.zeros(self.centroids.shape)
        for i in range(k):
            centroids[i] = np.mean(X[clustering == i], axis = 0)
        self.centroids = centroids


    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            # your code
            k = self.n_clusters
            centroids = np.zeros([k,X.shape[1]])
            for i in range(k):
                centroids[i] = X[np.random.randint(X.shape[0])] # random data point as the centroids
        
            self.centroids=centroids

        elif self.init == 'kmeans++':
            # your code
            k = self.n_clusters
            centroids = np.zeros([1, X.shape[1]])
            centroids[0] = X[np.random.randint(X.shape[0])]

            for i in range (k-1):
                dist = self.euclidean_distance(X, centroids)
                # get the each minimum distance of data points to all centroids, then get the index of which minimum distance is the largest
                c = np.argmax(np.min(dist,axis=1)) 
                centroids = np.vstack([centroids, X[c,:]])

            self.centroids=centroids

        else:
            raise ValueError('Centroid initialization method should either be "random" or "kmeans++"')


    # used method from this https://www.dabblingbadger.com/blog/2020/2/27/implementing-euclidean-distance-matrix-calculations-from-scratch-in-python
    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """euclidean_distance
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # your code
        X1_dots = np.sum(np.square(X1),axis=1)
        X2_dots = np.sum(np.square(X2),axis=1)
        dist = np.sqrt(abs(X1_dots.reshape(-1,1) + X2_dots - 2*np.dot(X1, X2.T)))
        return dist


    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # your code
        a = []
        b = []
        for i in range (X.shape[0]):
            # calculate the distance between the current data point and its cluster representative
            d1 = self.euclidean_distance(np.array([X[i,:]]), np.array([self.centroids[clustering[i],:]]))
            a.append(d1.reshape(-1)[0])
            # calculate the distance between the current data point and the "second best" cluster representative
            d2 = self.euclidean_distance(np.array([X[i,:]]), np.delete(self.centroids, clustering[i], axis=0))
            b.append(np.min(d2))
        a = np.array(a)
        b = np.array(b)
        denominator = np.maximum(a,b)
        coefficient = np.mean(np.divide((b-a), denominator, where= denominator !=0))
        return coefficient
