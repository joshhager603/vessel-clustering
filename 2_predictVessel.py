import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
import functools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

'''
======================================================================================================
-- README --
CSDS 340
Case Study 2 Python Code
Group 2
Sam King (sjk171), Calvin Cai (cyc44), Josh Hager (jrh236)

Before running the code, please ensure the files ./Data/set1.csv, ./Data/set2.csv, 
and ./Data/set3.csv exist and are in the proper directory.

The code can be run using the command:
python3 2_predictVessel.py
======================================================================================================
'''

def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))


def predictor_baseline(csv_path):
    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization 
    X = preprocessing.StandardScaler().fit(X).transform(X)
    # k-means with K = number of unique VIDs of set1
    K = 20
    model = KMeans(n_clusters=K, random_state=123, n_init='auto').fit(X)
    # predict cluster numbers of each sample
    labels_pred = model.predict(X)
    return labels_pred


def get_baseline_score():
    file_names = ['set1.csv', 'set2.csv']
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        labels_pred = predictor_baseline(csv_path)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Adjusted Rand Index Baseline Score of {file_name}: {rand_index_score:.4f}')

def baseline_preprocess(csv_path):
    '''
    Baseline preprocessing done for all models.
    '''

    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization 
    X = preprocessing.StandardScaler().fit(X).transform(X)

    return X

def predictor(csv_path):
    # k = 17 chosen as optimal number of clusters for set 3
    return agg_predictor(csv_path, 17)

def evaluate():
    csv_path = './Data/set3.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set3.csv: {rand_index_score:.4f}')

def num_unique_VIDs(csv_path):
    '''
    Counts the number of unique VIDs in a dataset.
    '''

    vids = pd.read_csv(csv_path)['VID'].to_list()

    num_unique = 0
    seen = set()

    for i in range(len(vids)):
        vid = vids[i]

        if vid not in seen:
            seen.add(vid)
            num_unique += 1

    return num_unique

def agg_clustering(csv_path):
    '''
    Performs grid search over chosen parameters for agglomerative clustering.
    '''

    results = []

    linkage = ['ward', 'complete', 'average', 'single']

    n_clusters = num_unique_VIDs(csv_path)

    X = baseline_preprocess(csv_path)

    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()

    # no pca, pass through all linkage types
    for i in range (0,4):
        labels_pred = AgglomerativeClustering(n_clusters = n_clusters, linkage = linkage[i]).fit_predict(X)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)

        results.append([
            rand_index_score,
            f"No PCA, linkage = {linkage[i]}"
            ])

    #pca, pass through all linkage types
    for i in range (0,4):
        for n_components in range(3,6):
            pca = PCA(n_components=n_components)
            X_transformed = pca.fit_transform(X)
            pred_labels = AgglomerativeClustering(n_clusters = n_clusters, linkage = linkage[i]).fit_predict(X_transformed)
            rand_index_score = adjusted_rand_score(labels_true, labels_pred)

            results.append([
                rand_index_score,
                f"No PCA, linkage = {linkage[i]}"
                ])
    return results

def grid_search_kmeans(csv_path):
    '''
    Perfoms grid search over chosen parameters for k-means clustering.
    '''

    results = []
    n_clusters = num_unique_VIDs(csv_path)

    X = baseline_preprocess(csv_path)

    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()

    # parameters to vary:
    # PCA or no PCA
    # PCA num components: 3-5
    # init: k-means++, random
    # n_init: 1-20
    # tol: 0.0001-0.0010

    ## k-means++, no PCA
    for n_init in range(1, 21):
        for tol in np.arange(0.0001, 0.0010, 0.0001):
            model = KMeans(n_clusters=n_clusters, random_state=123, init='k-means++', n_init=n_init, tol=tol).fit(X)
            labels_pred = model.predict(X)
            rand_index_score = adjusted_rand_score(labels_true, labels_pred)

            results.append([
                rand_index_score,
                f"No PCA, init='k-means++', n_init={n_init}, tol={tol}"
            ])

    ## random, no PCA
    for n_init in range(1, 21):
        for tol in np.arange(0.0001, 0.0010, 0.0001):
            model = KMeans(n_clusters=n_clusters, random_state=123, init='random', n_init=n_init, tol=tol).fit(X)
            labels_pred = model.predict(X)
            rand_index_score = adjusted_rand_score(labels_true, labels_pred)

            results.append([
                rand_index_score,
                f"No PCA, init='random', n_init={n_init}, tol={tol}"
            ])

    ## k-means++, PCA
    for n_init in range(1, 21):
        for tol in np.arange(0.0001, 0.0010, 0.0001):
            for n_components in range(3, 6):
                pca = PCA(n_components=n_components)
                X_transformed = pca.fit_transform(X)

                model = KMeans(n_clusters=n_clusters, random_state=123, init='k-means++', n_init=n_init, tol=tol).fit(X_transformed)
                labels_pred = model.predict(X_transformed)
                rand_index_score = adjusted_rand_score(labels_true, labels_pred)

                results.append([
                    rand_index_score,
                    f"PCA n_components={n_components}, init='k-means++', n_init={n_init}, tol={tol}"
                ])

    ## random, PCA
    for n_init in range(1, 21):
        for tol in np.arange(0.0001, 0.0010, 0.0001):
            for n_components in range(3, 6):
                pca = PCA(n_components=n_components)
                X_transformed = pca.fit_transform(X)

                model = KMeans(n_clusters=n_clusters, random_state=123, init='random', n_init=n_init, tol=tol).fit(X_transformed)
                labels_pred = model.predict(X_transformed)
                rand_index_score = adjusted_rand_score(labels_true, labels_pred)

                results.append([
                    rand_index_score,
                    f"PCA n_components={n_components}, init='random', n_init={n_init}, tol={tol}"
                ])

    return results

def avg_grid_search_results(set1, set2):
    '''
    Combines the grid search results for both set1.csv and set2.csv
    through a simple average.
    '''

    assert len(set1) == len(set2)

    avgset = []

    for i in range(len(set1)):
        avgset.append([
            # average the rand index scores, keep the parameters the same
            (set1[i][0] + set2[i][0]) / 2,
            set1[i][1]
        ])

    return avgset

def find_best_result(avgset):
    '''
    Searches through the averaged result set for the best rand index score
    and returns [score, parameters for that score].
    '''

    best_result = avgset[0]

    for i in range(len(avgset)):
        if avgset[i][0] > best_result[0]:
            best_result = avgset[i]

    return best_result

def k_means_tune():
    '''
    Performs hyperparameter tuning and feature extraction for k-means clustering.
    '''

    # grid search over chosen hyperparameters for both datasets
    set1 = grid_search_kmeans('./Data/set1.csv')
    set2 = grid_search_kmeans('./Data/set2.csv')

    # average the results of the two grid searches
    avgset = avg_grid_search_results(set1, set2)

    # find the parameters that led to the best rand index score
    best_result = find_best_result(avgset)

    return best_result

def agg_clustering_tune():
    '''
    Performs hyperparameter tuning and feature extraction for agglomerative clustering.
    '''

    # grid search over chosen parameters for both datasets
    set1 = agg_clustering('./Data/set1.csv')
    set2 = agg_clustering('./Data/set2.csv')

    # average the results of the two grid searches
    avgset = avg_grid_search_results(set1, set2)

    # find the parameters that led to the best rand index score
    best_result = find_best_result(avgset)

    return best_result

def k_means_predictor(csv_path, n_clusters):
    '''
    Runs k-means clustering on data in csv_path using n clusters.
    '''

    # perform the baseline preprocessing on the data
    X = baseline_preprocess(csv_path)

    # perform PCA feature extraction
    pca = PCA(n_components=4)
    X_transformed = pca.fit_transform(X)

    # train model using optimally found parameters
    # OPTIMAL: [0.3273459423370036, "PCA n_components=4, init='random', n_init=2, tol=0.0008"]
    model = KMeans(n_clusters=n_clusters, random_state=123, init='random', n_init=2, tol=0.0008).fit(X_transformed)

    # predict on the transformed data
    labels_pred = model.predict(X_transformed)

    return labels_pred

def agg_predictor(csv_path, n_clusters):

    # perform baseline preprocessing on the data
    X = baseline_preprocess(csv_path)

    # train model using optimally found parameters and fit to the data
    # OPTIMAL: [0.32550070202469833, 'No PCA, linkage = ward']
    labels_pred = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X)

    return labels_pred

def evaluate_general(csv_path, predictor):
    '''
    A more general form of the evaluate function that takes a csv_path and
    a predictor and applies that predictor to the data in the csv.
    '''

    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()

    n_clusters = num_unique_VIDs(csv_path) # use n_clusters equal to the number of unique VIDs
    labels_pred = predictor(csv_path, n_clusters)

    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of {csv_path}: {rand_index_score:.4f}')

def evaluate_noprint(csv_path, predictor):
    '''
    A more general form of the evaluate function that takes a csv_path and
    a predictor and applies that predictor to the data in the csv.

    Only returns the rand index score instead of printing it.
    '''

    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()

    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)

    return rand_index_score

def find_optimal_k_silhouette_std(csv_path):
    '''
    Graphs standard deviations of silhouette samples at each cluster.
    '''

    X = baseline_preprocess(csv_path)
    silhouette_std = []
    k_values = range(2, 30)
    
    for k in k_values:
        clf = AgglomerativeClustering(n_clusters=k, linkage='ward')
        predict = clf.fit_predict(X)
        silhouette_vals = silhouette_samples(X, predict)
        silhouette_std.append(np.std(silhouette_vals))
    
    plt.plot(k_values, silhouette_std, marker='o')
    plt.title('Standard Deviations of Silhouette Scores for Different Values of K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Standard Deviation of Silhouette Scores')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()
    

def find_optimal_k_silhouette_score(csv_path):
    '''
    Graphs average silhouette scores at each cluster.
    '''

    X = baseline_preprocess(csv_path)
    silhouette = []
    k_values = range(2, 30)
    
    for k in k_values:
        clf = AgglomerativeClustering(n_clusters=k, linkage='ward')
        predict = clf.fit_predict(X)
        silhouette_vals = silhouette_score(X, predict)
        silhouette.append(silhouette_vals)
    
    plt.plot(k_values, silhouette, marker='o')
    plt.title('Average Silhouette Scores for Different Values of K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Average Silhouette Scores')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()


def find_optimal_k_elbow(csv_path):
    '''
    Finds the optimal k using the elbow method.
    '''

    X = baseline_preprocess(csv_path)
    sse = []
    k_values = range(2, 30, 2)
    
    for k in k_values:
        clf = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clf.fit_predict(X)
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        sse.append(((X - centroids[labels])**2).sum())
    plt.plot(k_values, sse, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()
    
if __name__=="__main__":
    get_baseline_score()
    evaluate()