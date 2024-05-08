import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import functools
from sklearn.metrics.cluster import adjusted_rand_score

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


def evaluate():
    csv_path = './Data/set3.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set3.csv: {rand_index_score:.4f}')


def baseline_preprocess(csv_path):
    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization 
    X = preprocessing.StandardScaler().fit(X).transform(X)

    return X

def num_unique_VIDs(csv_path):
    vids = pd.read_csv(csv_path)['VID'].to_list()

    num_unique = 0
    seen = set()

    for i in range(len(vids)):
        vid = vids[i]

        if vid not in seen:
            seen.add(vid)
            num_unique += 1

    return num_unique


def grid_search_kmeans(csv_path):
    results = []
    n_clusters = num_unique_VIDs(csv_path)

    X = baseline_preprocess(csv_path)

    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()

    # PCA or no PCA
    # PCA num components: 1-5
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
    assert len(set1) == len(set2)

    avgset = []

    for i in range(len(set1)):
        avgset.append([
            (set1[i][0] + set2[i][0]) / 2,
            set1[i][1]
        ])

    return avgset

def find_best_result(avgset):
    best_result = avgset[0]

    for i in range(len(avgset)):
        if avgset[i][0] > best_result[0]:
            best_result = avgset[i]

    return best_result

def k_means_tune():
    set1 = grid_search_kmeans('./Data/set1.csv')
    set2 = grid_search_kmeans('./Data/set2.csv')

    avgset = avg_grid_search_results(set1, set2)

    best_result = find_best_result(avgset)

    return best_result

def k_means_predictor(csv_path, n_clusters):
    X = baseline_preprocess(csv_path)

    pca = PCA(n_components=4)
    X_transformed = pca.fit_transform(X)

    # OPTIMAL: [0.3273459423370036, "PCA n_components=4, init='random', n_init=2, tol=0.0008"]
    model = KMeans(n_clusters=n_clusters, random_state=123, init='random', n_init=2, tol=0.0008).fit(X_transformed)

    labels_pred = model.predict(X_transformed)

    return labels_pred

def evaluate_general(csv_path, predictor):
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()

    n_clusters = num_unique_VIDs(csv_path)
    labels_pred = predictor(csv_path, n_clusters)

    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of {csv_path}: {rand_index_score:.4f}')

def evaluate_noprint(csv_path, predictor):
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()

    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)

    return rand_index_score

def predictor(csv_path):
    # fill your code here
    #return labels_pred
    pass

get_baseline_score()
evaluate_general('./Data/set1.csv', k_means_predictor)
evaluate_general('./Data/set2.csv', k_means_predictor)

# if __name__=="__main__":
#     get_baseline_score()
#     evaluate()


