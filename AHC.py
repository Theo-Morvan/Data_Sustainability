import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
import ipdb


class Cluster:
    def __init__(self, X) -> None:
        if len(X.shape)==1:
            X = X.reshape((1,-1))
        self.X = X
        self.X_position = X[:,:-1]
        try:
            len(self.X_position.shape)==2
        except:
            raise(IndexError("X_position should be a 2-dim tensor"))
        self.flot = X[:,-1].sum()
        self.number_elements = X.shape[0]
        self.center = self.X_position.mean(axis=0)
        
    
    def compute_maximal_distance(self):
        distances = pdist(self.X_position)
        max_distances = distances.max()
        return max_distances
    
    def compute_maximal_distance_center(self):
        center_cluster = self.X_position.mean(axis=0)
        distances_from_center = (self.X_position[:,0] - center_cluster[0])**2 + (self.X_position[:,1]-center_cluster[1])**2 #on calcule euclidean distance par rapport à au centre du cluster, pas forcément prendre la moyen
        return distances_from_center.argmax(), distances_from_center.max()

class AgglomerativeHierarchicalClustering:

    def __init__(self, number_clusters:int, thresold_flot:int) -> None:
 
        self.number_clusters = number_clusters
        self.threshold_flot = thresold_flot

    def fit(self, X):
        clusters = []
        for i in range(len(X)):
            clusters.append(Cluster(X[i,:]))
        stop_algorithms = False

        for i in range(200):
            if len(clusters)<=self.number_clusters:
                self.clusters = clusters
                break
            clusters, stop_algorithms = self.perform_update(clusters)
            if stop_algorithms:
                break 
        self.final_clusters = clusters

    def compute_distance_between_clusters(self, cluster_1:Cluster, cluster_2:Cluster):
        try:
            matrix_distance = distance_matrix(cluster_1.X_position, cluster_2.X_position)
        except:
            ipdb.set_trace()
    
        average_distance = matrix_distance.sum()/(cluster_1.number_elements*cluster_2.number_elements)
        return average_distance

    def merge_clusters(self, cluster_1:Cluster, cluster_2:Cluster):
        new_X = np.concatenate((cluster_1.X, cluster_2.X))
        new_cluster = Cluster(new_X)
        return new_cluster
    
    
    def perform_update(self,clusters):
        break_point=False
        clusters_to_merge = self.find_closest_clusters(clusters)
        if clusters_to_merge:
            new_cluster = self.merge_clusters(clusters[clusters_to_merge[0]], clusters[clusters_to_merge[1]])
            del clusters[clusters_to_merge[0]]
            del clusters[clusters_to_merge[1]]
            clusters.append(new_cluster)
        else:
            break_point=True
        return clusters, break_point

    def find_closest_clusters(self,clusters):
        min_distance = np.inf
        for i, cluster_1 in enumerate(clusters):
            if cluster_1.flot > self.threshold_flot:
                pass
            else:
                for j, cluster_2 in enumerate(clusters[i+1:]):
                    if cluster_2.flot > self.threshold_flot:
                        pass
                    else:
                        cluster_distance = self.compute_distance_between_clusters(cluster_1, cluster_2)
                        if cluster_distance <min_distance:
                            clusters_to_merge = (i,j)
                            min_distance = cluster_distance
        try:                   
            return clusters_to_merge
        except:
            return None


if __name__=="__main__":
    X = np.random.randn(100,3)
    X[:,-1] = np.random.randint(low=30, high=1000, size=(100,))
    ahc = AgglomerativeHierarchicalClustering(3, thresold_flot=6000)
    ahc.fit(X)
    ipdb.set_trace()