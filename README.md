# BFR-Algorithm
Implemented BFR Algorithm for clustering high Dimensional Data

BFR Algorithm: A variant of k-means to handle large datasets, it assumes clusters are normally distributed around a centroid in a Euclidean space. It uses Mahalanobis distance as distance measure
- Requires O(cluster) memory instead of O(Data) as it uses summary statistics of groups of points instead of storing points
- It stores data as 3 sets: Cluster summaries, Outliers, Points to be clustered

Algorithm Overview:
1. Initialize K centroids
2. Load in a bag of points from disk
3. Assign new points to one of K original clusters, if within some distance threshold of the cluster
4. create new clusters with remaining points
5. Merge new clusters from step 4 with any of the existing clusters
6. Repeat steps 2-5 until all points are examined


![alt BFR](https://github.com/madamalarevanth/BFR-Algorithm/blob/main/BFR.jpeg)
