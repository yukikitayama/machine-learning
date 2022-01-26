## Clustering

- `Centroid` is a data point representing the center of a cluster.
    - In each kth cluster, take mean in each feature.
- `from sklearn.datasets import make_blobs, make_moons` are convenient functions to generate synthetic clusters.

## K-Means Clustering

- Pre-specify the number of clusters `K`.
- Minimize the sum of the within-cluster variations over all K clusters.
- The most common choice for the within-cluster variation is to use `squared Euclidean distance`.
    - The sum of all the pairwise squared Euclidean distance within kth cluster, divided by the number of data in kth
      cluster.
- K-mean clustering is called `nondeterministic`, meaning cluster assignment could change depending on the random
  initialization.
    - Commonly run several initializations of the entire k-means algorithm and find the lowest error.
    - By default, `scikit-learn` runs k-means clustering 10 times, and return the one with the lowest error.

### Elbow method

- Choose `K` by seeing the reduction of residual sum of squares

### Silhouette Coefficient

- Values range between -1 and 1.
- Larger numbers indicate samples are closer to their clusters than they are to other clusters.
    - It doesn't work if the clusters are nonspherical.
- `Silhouette coefficient = (b - a) / max(a, b)`
    - `a`: The mean distance between a sample and all other points in the same class.
    - `b`: The mean distance between a sample and all other points in the next nearest cluster.
- `scikit-learn` takes average of silhouette coefficients of all samples into one score
    - Pick the `K` with the max silhouette coefficient.
- [scikit-learn 2.3.10.5. Silhouette Coefficient](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)

### Adjusted Rand Index (ARI)

- Evaluate clustering by using both predicted cluster label and true cluster label.
- `sklearn.metrics.adjusted_rand_score` returns close to 0.0 when labeling is random and close to 1.0 when the
  clusterings are identical.

## DBSCAN

- Density-Based Spatial Clustering of Applications with Noise

## Hierarchical Clustering

- Algorithm
    - Normalize features to make them equally important
    - Make a `dendrogram`, the higher the height, the more the data are different. Horizontal cut defines clusters.
    - Treat each data as its own cluster
    - Fuse the two clusters that are most similar to each other
        - `Euclidean distance` is commonly used to measure similarity.
        - `Linkage` is used within a cluster to calculate single score for similarity.
            - `Average linkage` calculates all pairwise similarity scores within a cluster and take average
            - `Complete linkage` calcuates all pairwise similarity scores and take the largest score.
            - Average and complete linkage tend to yield more balanced clusters.
    - Repeat until all the data belong to one single cluster.
- If the true clusters are not nested, hierarchical clustering could not well represent clusters.
- `Agglomerative clustering`
    - Agglomerate means to collect or gather into a cluster or mass.

### Resource

- [scikit-learn hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
