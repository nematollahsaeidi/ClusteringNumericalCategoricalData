import time
import csv
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import cluster, mixture, metrics
from sklearn.neighbors import kneighbors_graph
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder

def get_support(data, feature_id, feature_val, cluster):
    """Compute support for a given value in a cluster."""
    return sum(data[cluster[j], feature_id] == feature_val for j in range(len(cluster)))

def similarity_instance_cluster(data, instance_id, cluster):
    """Compute similarity between an instance and a specified cluster."""
    similarity = 0.0
    n_features = data.shape[1]

    for feature_id in range(n_features):
        unique_values = {data[cluster[j], feature_id] for j in range(len(cluster))}
        total_support = sum(get_support(data, feature_id, val, cluster) for val in unique_values)
        similarity += get_support(data, feature_id, data[instance_id, feature_id], cluster) / total_support

    return similarity

def squeezer(data, threshold):
    """Squeezer algorithm for clustering categorical data."""
    clusters = [[0]]

    for instance_id in range(1, data.shape[0]):
        similarities = [similarity_instance_cluster(data, instance_id, cluster) for cluster in clusters]
        max_similarity = max(similarities)
        best_cluster_idx = similarities.index(max_similarity)

        if max_similarity >= threshold:
            clusters[best_cluster_idx].append(instance_id)
        else:
            clusters.append([instance_id])

    return clusters

def squeezer_cluster(data, threshold=7):
    """Run Squeezer clustering and return results."""
    data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    result = squeezer(data, threshold)
    print(f'Number of clusters: {len(result)}')
    return result, len(result)

def kprototypes_cluster(data, categorical, n_clusters=5, init='Cao', verbose=1, n_init=10):
    """Run k-prototypes clustering."""
    from kmodes.kprototypes import KPrototypes

    kp = KPrototypes(n_clusters=n_clusters, init=init, verbose=verbose, n_init=n_init)
    clusters = kp.fit_predict(data, categorical=categorical)
    return kp, clusters

def numeric_cluster(X):
    """Cluster numerical data using various algorithms."""
    df_cluster = pd.DataFrame()
    X = StandardScaler().fit_transform(X)

    params = {
        'quantile': 0.1,
        'eps': 0.01,
        'damping': 0.77,
        'preference': -20,
        'n_neighbors': 2,
        'n_clusters': 15,
        'min_samples': 1,
        'xi': 0.25,
        'min_cluster_size': 0.01
    }

    connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)

    clustering_algorithms = [
        ('MiniBatchKMeans', cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])),
        ('AffinityPropagation', cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])),
        ('MeanShift', cluster.MeanShift(bandwidth=cluster.estimate_bandwidth(X, quantile=params['quantile']), bin_seeding=True)),
        ('SpectralClustering', cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")),
        ('Ward', cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)),
        ('AgglomerativeClustering', cluster.AgglomerativeClustering(linkage="complete", affinity="cityblock", n_clusters=params['n_clusters'], connectivity=connectivity)),
        ('DBSCAN', cluster.DBSCAN(eps=params['eps'])),
        ('OPTICS', cluster.OPTICS(min_samples=params['min_samples'], xi=params['xi'], min_cluster_size=params['min_cluster_size'])),
        ('Birch', cluster.Birch(n_clusters=params['n_clusters'])),
        ('GaussianMixture', mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full'))
    ]

    for name, algorithm in clustering_algorithms:
        try:
            start_time = time.time()
            algorithm.fit(X)
            labels = algorithm.labels_ if hasattr(algorithm, 'labels_') else algorithm.predict(X)
            df_cluster[name] = labels
            silhouette = metrics.silhouette_score(X, labels)
            elapsed_time = time.time() - start_time
            print(f'{name} silhouette_score: {silhouette:.4f}, time: {elapsed_time:.4f}s')
        except Exception as e:
            print(f'{name} skipped: {e}')

    return df_cluster

def csv_to_tsv(input_file, output_file='/content/NAD.txt'):
    """Convert a CSV file to a TSV file."""
    with open(input_file, 'r') as csvin, open(output_file, 'w') as tsvout:
        reader = csv.reader(csvin)
        writer = csv.writer(tsvout, delimiter='\t')
        writer.writerows(reader)
    print(f'File saved to {output_file}')

def cat_utility(ds, clustering, m):
    n = len(ds)
    d = len(ds[0])

    cluster_cts = [0] * m
    for ni in range(n):
        k = clustering[ni]
        cluster_cts[k] += 1

    for i in range(m): 
        if cluster_cts[i] == 0:
            return 0.0

    unique_vals = [0] * d
    for i in range(d):
        maxi = 0
        for ni in range(n):
            if ds[ni][i] > maxi: maxi = ds[ni][i]
        unique_vals[i] = maxi + 1

    att_cts = []
    for i in range(d):
        cts = [0] * unique_vals[i] 
        for ni in range(n):
            v = ds[ni][i]
            cts[v] += 1
        att_cts.append(cts)

    k_cts = []
    for k in range(m):
        a_cts = []
        for i in range(d):
            cts = [0] * unique_vals[i] 
            for ni in range(n):
                if clustering[ni] != k: continue
                v = ds[ni][i]
                cts[v] += 1
            a_cts.append(cts)
        k_cts.append(a_cts)

    un_sum_sq = 0.0 
    for i in range(d):  
        for j in range(len(att_cts[i])):
            un_sum_sq += (1.0 * att_cts[i][j] / n) * (1.0 * att_cts[i][j] / n)

    cond_sum_sq = [0.0] * m  
    for k in range(m):
        sum = 0.0
        for i in range(d):
            for j in range(len(att_cts[i])):
                if cluster_cts[k] == 0: print("FATAL LOGIC ERROR")
                sum += (1.0 * k_cts[k][i][j] / cluster_cts[k]) * (1.0 * k_cts[k][i][j] / cluster_cts[k])
        cond_sum_sq[k] = sum

    prob_c = [0.0] * m
    for k in range(m):
        prob_c[k] = (1.0 * cluster_cts[k]) / n
  
    left = 1.0 / m
    right = 0.0
    for k in range(m):
        right += prob_c[k] * (cond_sum_sq[k] - un_sum_sq)
    cu = left * right
    return cu

def cluster(ds, m):
    n = len(ds)

    working_set = [0] * m
    for k in range(m):
        working_set[k] = list(ds[k]) 
    
    clustering = list(range(m))

    for i in range(m, n):
        item_to_cluster = ds[i]
        working_set.append(item_to_cluster)

        proposed_clusterings = []
        for k in range(m):
            copy_of_clustering = list(clustering) 
            copy_of_clustering.append(k)
            proposed_clusterings.append(copy_of_clustering) 

        proposed_cus = [0.0] * m
        for k in range(m):
            proposed_cus[k] = cat_utility(working_set, proposed_clusterings[k], m)

        best_proposed = np.argmax(proposed_cus)
        clustering.append(best_proposed)

    return clustering

def category_utility(raw_data, clustering, m=3):
    print("\nBegin clustering using category utility demo")

    raw_data = [['red  ','short ','heavy'],
                ['blue ','medium','heavy'],
                ['green','medium','heavy'],
                ['red  ','long  ','light'],
                ['green','medium','light']]
  
    data = np.array(raw_data)
    raw_data1 = pd.DataFrame(data)
    le = LabelEncoder()
    enc_data = raw_data1.apply(le.fit_transform)
    enc_data = np.array(enc_data)

    print("\nRaw data:")
    for item in raw_data:
        print(item)

    print("\nEncoded data:")
    for item in enc_data:
        print(item)
  
    print(f"\nStart clustering with m = {m}")
    clustering = cluster(enc_data, m)
    print("Done")

    print("\nResult clustering:")
    clustering = np.array([0, 1, 2, 0, 2])

    cu = cat_utility(enc_data, clustering, m)
    print(f"Category utility of clustering = {cu:.4f}\n")

    print("\nClustered raw data:")
    for k in range(m):
        for i in range(len(enc_data)):
            if clustering[i] == k:
                print(raw_data[i])

    print("\nEnd demo\n")
    
    return cu

def VAR(raw_data, clustering):
    raw_data = [[1,2,3],
                [1,2,4],
                [7,8,9],
                [10,8,9],
                [20,94,15]]
    clustering = np.array([0, 0, 1, 1, 2])

    cou = Counter()
    for i in clustering:
        cou[i] += 1
    print(f'Cluster counts: {cou}')

    clusters = defaultdict(list)
    for idx, cluster_id in enumerate(clustering):
        clusters[cluster_id].append(raw_data[idx])

    var1 = 0
    for i in range(len(set(clustering))):
        var1 += (np.var(clusters[i], axis=0)) / cou[i]
    
    var_all = np.sqrt(np.dot(var1, var1))
    print(f'Variance: {var_all}')

    return var_all

def CV(CU, Variance):
    """Compute CV according to CU and Variance."""
    CV = CU / (1 + Variance)
    print(f'CV: {CV}')
    return CV