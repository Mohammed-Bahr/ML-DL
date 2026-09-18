import numpy as np
import matplotlib.pyplot as plt


class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))  # axis=1 fixes the bug

    def fit(self, X, max_iterations=200):
        self.centroids = np.random.uniform(
            np.amin(X, axis=0),
            np.amax(X, axis=0),
            size=(self.k, X.shape[1]),
        )

        for _ in range(max_iterations):
            y = []
            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            y = np.array(y)

            cluster_centers = []
            for i in range(self.k):
                points = X[y == i]
                if len(points) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(points.mean(axis=0))

            cluster_centers = np.array(cluster_centers)

            if np.allclose(self.centroids, cluster_centers):
                break

            self.centroids = cluster_centers

        return y


# Demo
random_points = np.random.randint(0, 100, (100, 2))

kmeans = KMeansClustering(k=3)
labels = kmeans.fit(random_points)

plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
            c=range(len(kmeans.centroids)), marker="*", s=200)
plt.show()
