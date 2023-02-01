"""
Drift detection algorithm from
[2] Y. Yuan, Z. Wang, and W. Wang,
“Unsupervised concept drift detection based on multi-scale slide windows,”
Ad Hoc Networks, vol. 111, p. 102325, Feb. 2021, doi: 10.1016/j.adhoc.2020.102325.

@author: Jindrich POHL

MSSW is an abbreviation for Multi-Scale Sliding Windows

- Unless specified otherwise, functions in this file work with numpy arrays
- The terms "benchmark data" and "reference data" mean the same thing, default is "reference data"
- The terms "slide data" and "testing data" mean the same thing, default is "testing data"
"""
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.io import arff
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def ptg_for_all(reference_data):
    """
    Calculate all P_tgs from reference data

    :param reference_data: array of shape (#points, #attributes) of reference data
    :return: array of shape (#points, #attribute) of corresponding P_tgs
    """
    column_sum = np.sum(reference_data, axis=0)
    return np.divide(reference_data, column_sum)


def information_utilities_for_all(ptgs):
    """
    Calculate information utility values from P_tgs

    :param ptgs: P_tgs as obtained from ptg_for_all(...)
    :return: array of shape (1, #attributes) of the information utility of each attribute
    """
    entropies = np.divide(scipy.stats.entropy(ptgs, axis=0), np.log(ptgs.shape[0]))
    entropies = np.where(entropies > 1, 1.0, entropies)
    information_utilities = np.subtract(1, entropies).reshape((1, entropies.shape[0]))
    return information_utilities


def attribute_weights_for_all(information_utilities):
    """
    Calculate the weights of attributes from information utilities

    :param information_utilities: information utilities as obtained from information_utilities_for_all(...)
    :return: array of shape (1, #attributes) of the attribute weights of each attribute
    """
    attribute_weights = np.divide(information_utilities, np.sum(information_utilities))
    return attribute_weights


def get_attribute_weights_from(reference_data):
    """
    Calculate weights of attributes from reference (benchmark) data

    :param reference_data: array of shape (#points, #attributes)
    :return: array of shape (1, #attributes) of the attribute weights of each attribute
    """
    ptgs = ptg_for_all(reference_data)
    information_utilities = information_utilities_for_all(ptgs)
    attribute_weights = attribute_weights_for_all(information_utilities)
    return attribute_weights


def transform_data_by_attribute_weights(original_data, attribute_weights):
    """
    Transform data by the sqrt of attribute weights

    :param original_data: array of shape (#points, #attributes) to transform
    :param attribute_weights: array of shape (1, #attributes) to use for the transformation
    :return: array of shape (#points, #attributes) of weighted data
    """
    sqrt_attribute_weights = np.sqrt(attribute_weights)
    weighted_data = np.multiply(original_data, sqrt_attribute_weights)
    return weighted_data


def transform_batches_by_attribute_weights(original_batches, attribute_weights):
    """
    Transform multiple batches of data by the sqrt of attribute weights

    :param original_batches: list of arrays of shape (n_i, #attributes), i=batch number, n_i > 1
    :param attribute_weights: array of shape (1, #attributes) of weights to use for the transformation
    :return: list of arrays of shape(n_i, #attributes) of weighted data
    """
    weighted_batches = []
    for original_batch in original_batches:
        weighted_batches.append(transform_data_by_attribute_weights(original_batch, attribute_weights))
    return weighted_batches


def mssw_preprocess(reference_data_batches, testing_data_batches):
    """
    Preprocess data batches through minmax scaling, apply weighting so that Euclidean distance on this weighted data
    becomes the desired entropy-weighted distance on the original data

    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :return: (array of shape (sum(n_r_r) #attributes) of joined reference data, weighted reference batches (same
        structure as reference_data_batches), weighted testing batches (same structure as testing_data_batches))
    """
    joined_reference_data = reference_data_batches[0]
    for reference_batch in reference_data_batches[1:]:
        np.append(joined_reference_data, reference_batch, axis=0)

    scaler = MinMaxScaler()
    scaler.fit(joined_reference_data)
    joined_reference_data = scaler.transform(joined_reference_data)
    reference_data_batches = [scaler.transform(batch) for batch in reference_data_batches]
    testing_data_batches = [scaler.transform(batch) for batch in testing_data_batches]

    small_float = np.finfo(dtype=float).eps * (10 ** 6)
    joined_reference_data = np.where(joined_reference_data == 0, small_float, joined_reference_data)
    reference_data_batches = [np.where(batch == 0, small_float, batch) for batch in reference_data_batches]
    testing_data_batches = [np.where(batch == 0, small_float, batch) for batch in testing_data_batches]

    attribute_weights = get_attribute_weights_from(joined_reference_data)
    weighted_joined_reference_data = transform_data_by_attribute_weights(joined_reference_data, attribute_weights)
    weighted_reference_batches =\
        [transform_data_by_attribute_weights(batch, attribute_weights) for batch in reference_data_batches]
    weighted_testing_batches =\
        [transform_data_by_attribute_weights(batch, attribute_weights) for batch in testing_data_batches]
    return weighted_joined_reference_data, weighted_reference_batches, weighted_testing_batches





def obtain_cluster_distances_and_sizes(weighted_sub_window, fitted_kmeans, n_clusters):
    """
    Get the sum of centroid distances and size for clusters formed by fitted_kmeans and weighted_sub_window

    :param weighted_sub_window: array of shape (#points, #attributes) of weighted data
    :param fitted_kmeans: fitted sklearn kmeans object to use for clustering of the weighted_sub_window
    :param n_clusters: number of clusters used to fit the kmeans object
    :return: (array of shape (1, n_clusters) of sums of centroid distances,
    array of shape (1, n_clusters) of cluster sizes)
    """
    centroids = fitted_kmeans.cluster_centers_
    predicted_cluster_labels = fitted_kmeans.predict(weighted_sub_window)

    centroid_distance_sums = np.zeros(n_clusters).reshape((1, n_clusters))
    num_points_in_clusters = np.zeros(n_clusters).reshape((1, n_clusters))
    for cluster_id in range(n_clusters):
        cluster_mask = predicted_cluster_labels == cluster_id
        cluster = weighted_sub_window[cluster_mask]

        num_points_in_clusters[0, cluster_id] = cluster.shape[0]

        centroid = centroids[cluster_id]
        centroid_diffs = np.subtract(cluster, centroid)
        euclideans = np.linalg.norm(centroid_diffs, axis=1)
        sum_euclideans = np.sum(euclideans)
        centroid_distance_sums[0, cluster_id] = sum_euclideans

    return centroid_distance_sums, num_points_in_clusters


def calculate_clustering_statistics(weighted_sub_window, fitted_kmeans, n_clusters):
    """
    Cluster the given weighted_sub_window, and then obtain JSEE, Av_ci for all i, and Av_sr from it

    :param weighted_sub_window: array of shape (#points, #attributes) of weighted data
    :param fitted_kmeans: fitted sklearn kmeans object to use for clustering of the weighted_sub_window
    :param n_clusters: number of clusters used to fit the kmeans object
    :return: (JSEE float, Av_ci array of shape (1, n_clusters), Av_sr float,
    num_points_in_clusters array of shape (1, n_clusters) of sizes of clusters)
    """
    centroid_distance_sums, num_points_in_clusters = obtain_cluster_distances_and_sizes(
        weighted_sub_window, fitted_kmeans, n_clusters
    )

    JSEE = np.sum(centroid_distance_sums)
    num_points_in_clusters = np.where(num_points_in_clusters == 0, 1, num_points_in_clusters)
    Av_c = np.divide(centroid_distance_sums, num_points_in_clusters)
    Av_sr = JSEE / weighted_sub_window.shape[0]
    return JSEE, Av_c, Av_sr, num_points_in_clusters


def get_s_s(weighted_reference_sub_windows, fitted_kmeans, n_clusters):
    """
    Get S_s = the total average distance sequence of sub-windows in reference (benchmark) data

    :param weighted_reference_sub_windows: list of arrays of shape (n_r, #attributes) of weighted reference data,
        r is the sub-window number, n_r=#points in this sub-window
    :param fitted_kmeans: sklearn kmeans object previously fitted to weighted reference (benchmark) data
    :param n_clusters: number of clusters used to fit the kmeans object
    :return: (s_s array of shape (1, #reference sub-windows)),
    all_av_c array of shape (n_clusters, len(weighted_reference_sub_windows) of Av_ci in each sub-window,
    all_cluster_num_points array of shape (n_clusters, #reference sub-windows)
    of #points in each cluster and sub-window)
    """
    num_sub_windows = len(weighted_reference_sub_windows)
    s_s = np.zeros(num_sub_windows).reshape((1, num_sub_windows))
    all_av_c = np.zeros(num_sub_windows * n_clusters).reshape((n_clusters, num_sub_windows))
    all_cluster_num_points = np.zeros(num_sub_windows * n_clusters).reshape((n_clusters, num_sub_windows))
    for i, weighted_reference_sub_window in enumerate(weighted_reference_sub_windows):
        _, Av_c, Av_sr, num_points_in_clusters =\
            calculate_clustering_statistics(weighted_reference_sub_window, fitted_kmeans, n_clusters)
        s_s[0, i] = Av_sr
        all_av_c[:, i:(i + 1)] = Av_c.T
        all_cluster_num_points[:, i:(i + 1)] = num_points_in_clusters.T
    return s_s, all_av_c, all_cluster_num_points


def get_moving_ranges(s_s):
    """
    Get moving ranges (MR_i) for each sub-window from S_s

    :param s_s: s_s as obtained from get_s_s(...)
    :return: moving_ranges array of shape (1, len(s_s)-1)
    """
    moving_ranges = np.abs(np.subtract(s_s[:, 1:], s_s[:, :-1]))
    return moving_ranges


def get_mean_s_s_and_mean_moving_ranges(weighted_reference_sub_windows, fitted_kmeans, n_clusters):
    """
    Find the S_s and MR sequences and return all their information

    :param weighted_reference_sub_windows: list of arrays of shape (n_r, #attributes) of weighted reference data,
        r is the sub-window number, n_r=#points in this sub-window
    :param fitted_kmeans: sklearn kmeans object previously fitted to weighted reference (benchmark) data
    :param n_clusters: number of clusters used to fit the kmeans object
    :return: (mean of S_s as float, mean of MR as float, s_s, all_av_c, all_cluster_num_points)
    """
    s_s, all_av_c, all_cluster_num_points = get_s_s(weighted_reference_sub_windows, fitted_kmeans, n_clusters)
    moving_ranges = get_moving_ranges(s_s)
    return np.mean(s_s), np.mean(moving_ranges), s_s, all_av_c, all_cluster_num_points


# - function to test for concept drift based on the total average distance from one testing (slide) sub-window
def concept_drift_detected(mean_av_s, mean_mr, weighted_testing_sub_window, fitted_kmeans, n_clusters, coeff):
    """
    Test for concept drift in one weighted testing sub-window and return all associated information

    :param mean_av_s: mean_s_s as obtained from get_mean_s_s_and_mean_moving_ranges(...)
    :param mean_mr: mean_mr as obtained from get_mean_s_s_and_mean_moving_ranges(...)
    :param weighted_testing_sub_window: array of shape (#points, #attributes) of one weighted testing sub-window
    :param fitted_kmeans: sklearn kmeans object previously fitted to weighted reference (benchmark) data
    :param n_clusters: number of clusters used to fit the kmeans object
    :param coeff: drift detection coefficient
    :return: (True if drift is detected and False otherwise,
    LCL_Av_s float of average centroid distance lower bound,
    UCL_Av_s float of average centroid distance upper bound,
    test_all_av_c array of all Av_ci in each testing batch,
    test_Av_sr array of Av_sr of each testing batch,
    num_points_in_clusters array of numbers of points in clusters in each testing window
    """
    UCL_Av_s = mean_av_s + coeff * mean_mr
    LCL_Av_s = mean_av_s - coeff * mean_mr
    _, test_all_av_c, test_Av_sr, num_points_in_clusters =\
        calculate_clustering_statistics(weighted_testing_sub_window, fitted_kmeans, n_clusters)

    return not (LCL_Av_s < test_Av_sr < UCL_Av_s), LCL_Av_s, UCL_Av_s, test_all_av_c, test_Av_sr, num_points_in_clusters


def all_drifting_batches(
        reference_data_batches,
        testing_data_batches,
        n_clusters=2,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        coeff=2.66
):
    """
    Find all drift locations based on the given reference and testing batches

    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :param n_clusters: desired number of clusters for kmeans
    :param n_init: desired n_init for scikit-learn's k-means
    :param max_iter: desired max_iter for scikit-learn's k-means
    :param tol: desired tol for scikit-learn's k-means
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :param coeff: coeff used to detect drift, default=2.66
    :return: a boolean list, length=len(testing_data_batches),
        an entry is True if drift was detected there and False otherwise
    """

    drifts_detected, _, _, _, _, _ = all_drifting_batches_return_plot_data(
        reference_data_batches=reference_data_batches,
        testing_data_batches=testing_data_batches,
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        coeff=coeff
    )

    return drifts_detected


def all_drifting_batches_return_plot_data(
        reference_data_batches,
        testing_data_batches,
        n_clusters=2,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        coeff=2.66
):
    """
    Find all drift locations based on the given reference and testing batches, return all associated information

    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :param n_clusters: desired number of clusters for k-means
    :param n_init: desired n_init for scikit-learn's k-means
    :param max_iter: desired max_iter for scikit-learn's k-means
    :param tol: desired tol for scikit-learn's k-means
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :param coeff: coeff used to detect drift, default=2.66
    :return: (drifts_detected boolean list where each entry corresponds to a drift decision in one testing batch,
    LCL_Av_s float of average centroid distance lower bound,
    UCL_Av_s float of average centroid distance upper bound,
    all_av_c array of all Av_ci in each batch,
    Av_sr array of Av_sr of each batch,
    num_points_in_clusters array of numbers of points in clusters in each batch
    """
    weighted_joined_reference_data, weighted_reference_batches, weighted_testing_batches =\
        mssw_preprocess(reference_data_batches, testing_data_batches)

    fitted_kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    ).fit(weighted_joined_reference_data)

    all_cluster_num_points = np.zeros(
        (len(reference_data_batches) + len(testing_data_batches)) * n_clusters).reshape(
        (n_clusters, len(reference_data_batches) + len(testing_data_batches)))
    all_av_sr = np.zeros(
        (len(reference_data_batches) + len(testing_data_batches))).reshape(
        (1, len(reference_data_batches) + len(testing_data_batches)))

    mean_av_s, mean_mr, s_s, all_av_c, all_cluster_num_points_ref =\
        get_mean_s_s_and_mean_moving_ranges(weighted_reference_batches, fitted_kmeans, n_clusters)
    all_av_sr[:, :len(reference_data_batches)] = s_s
    all_cluster_num_points[:, :len(reference_data_batches)] = all_cluster_num_points_ref

    drifts_detected = []
    for i, weighted_testing_batch in enumerate(weighted_testing_batches):
        drift_detected, LCL_Av_s, UCL_Av_s, test_all_av_c, test_av_sr, num_points_in_clusters_test =\
            concept_drift_detected(mean_av_s, mean_mr, weighted_testing_batch, fitted_kmeans, n_clusters, coeff)
        drifts_detected.append(drift_detected)
        all_av_c = np.hstack((all_av_c, test_all_av_c.T))
        all_av_sr[0, len(reference_data_batches) + i] = test_av_sr
        all_cluster_num_points[:, i + len(reference_data_batches)] = num_points_in_clusters_test

    return drifts_detected, LCL_Av_s, UCL_Av_s, all_av_c, all_av_sr, all_cluster_num_points




def verify_df_type(data, name):
    if not isinstance(data, pd.DataFrame):
        raise ValueError('All input data must be dataframes, but ' + name + ' is of type ' + type(data).__name__)
    return


def verify_df_shapes(ref_data, test_data, ref_labels):
    if len(ref_data.columns) != len(test_data.columns):
        raise ValueError('Input ref_data and test_data should have the same number of columns, but ref_data has '
                         + len(ref_data.columns).__str__() + ' columns, while test_data has '
                         + len(test_data.columns).__str__() + ' columns')
    if ref_labels is not None:
        if len(ref_labels.columns) != 1:
            raise ValueError('Input ref_labels should be a dataframe with a single column representing reference labels'
                             + ', but ' + len(ref_labels.columns).__str__() + ' columns were found')
        if ref_data.shape[0] != ref_labels.shape[0]:
            raise ValueError('Input ref_data and ref_labels should have the same number of rows, but ref_data has '
                             + ref_data.shape[0].__str__() + ' rows, while ref_labels has '
                             + ref_labels.shape[0].__str__() + ' rows')
    if ref_data.shape[0] < test_data.shape[0]:
        raise ValueError('test_data must be <= ref_data, but the given test_data has ' + test_data.shape[0].__str__()
                         + ' data points and the given ref_data has ' + ref_data.shape[0].__str__() + ' data points')
    return


def verify_input_dfs(ref_data, test_data, ref_labels):
    verify_df_type(ref_data, 'ref_data')
    verify_df_type(test_data, 'test_data')
    if ref_labels is not None:
        verify_df_type(ref_labels, 'ref_labels')
    verify_df_shapes(ref_data, test_data, ref_labels)


def check_inputs(ref_data, test_data, ref_labels, encoder, scaler):
    verify_input_dfs(ref_data, test_data, ref_labels)
    if scaler is None:
        raise ValueError('MSSW always requires minmax scaling, so scaler cannot be None')
    if scaler != 'minmax':
        raise ValueError('MSSW always requires minmax scaling, so the given scaler: ' + scaler.__str__()
                         + ' is unsupported')
    if encoder == 'target' and ref_labels is None:
        raise ValueError('With target encoding, ref_labels are required, but the given ref_labels are None')
    if ref_labels is not None and ref_labels.iloc[:, 0].nunique() > 2:
        raise ValueError('Only two-class classification is supported, but ref_labels has '
                         + ref_labels.iloc[:, 0].nunique().__str__() + ' unique values, and '
                         + ref_labels.iloc[:, 0].nunique().__str__() + ' > 2')


def divide_numeric_categorical(df_x):
    df_x_numeric = df_x.select_dtypes(include=[np.number])
    df_x_categorical = df_x.select_dtypes(exclude=[np.number])
    return df_x_numeric, df_x_categorical


def transform_and_join(ref_num, ref_cat, test_num, test_cat, fitted_encoder):
    ref_index = ref_cat.index
    test_index = test_cat.index
    ref_cat_transformed = pd.DataFrame(fitted_encoder.transform(ref_cat))
    test_cat_transformed = pd.DataFrame(fitted_encoder.transform(test_cat))
    ref_cat_transformed.set_index(ref_index, inplace=True)
    test_cat_transformed.set_index(test_index, inplace=True)
    ref_encoded = ref_num.join(ref_cat_transformed, lsuffix='_num').to_numpy()
    test_encoded = test_num.join(test_cat_transformed, lsuffix='_num').to_numpy()

    return ref_encoded, test_encoded


def encode_and_make_numpy(ref_data, test_data, ref_labels, encoder):
    ref_num, ref_cat = divide_numeric_categorical(ref_data)
    test_num, test_cat = divide_numeric_categorical(test_data)
    ref_encoded = ref_num.to_numpy()
    test_encoded = test_num.to_numpy()
    if encoder is not None:
        if encoder == 'onehot':
            encoder = OneHotEncoder(sparse=False)
            encoder.fit(ref_cat)
            ref_encoded, test_encoded = transform_and_join(ref_num, ref_cat, test_num, test_cat, fitted_encoder=encoder)
        elif encoder == 'target':
            encoder = TargetEncoder()
            encoder.fit(ref_cat, ref_labels)
            ref_encoded, test_encoded = transform_and_join(ref_num, ref_cat, test_num, test_cat, fitted_encoder=encoder)
        else:
            raise ValueError('Unsupported encoder: ' + encoder)

    return ref_encoded, test_encoded


def split_to_fixed_size_batches(array, batch_size):
    """Split array to batches of the given batch_size"""
    chunk_size = batch_size
    array_perfect_num_rows = array.shape[0] - (array.shape[0] % chunk_size)
    num_chunks = array_perfect_num_rows // chunk_size
    array_perfect_size = array[:array_perfect_num_rows, :]
    array_batches = np.array_split(array_perfect_size, num_chunks)

    return array_batches


def mssw(ref_data, test_data, ref_labels=None, encoder=None, scaler='minmax',
         n_clusters=8, n_init=100, max_iter=1000, tol=0, random_state=None, coeff=2.66):
    """
    Determine whether test_data is drifting by comparing it to ref_data

    :param ref_data: dataframe of shape (#reference points, #features)
    :param test_data: dataframe of shape (#testing points, #features)
    :param ref_labels: optional dataframe with class labels of shape (#reference points, 1)
    :param encoder: None if categorical variables should be excluded, 'onehot' or 'target' if you wish to encode them
    :param scaler: always 'minmax', only present here for API consistency
    :param n_clusters: n_clusters parameter to use in sklearn.cluster.KMeans()
    :param n_init: n_init parameter to use in sklearn.cluster.KMeans()
    :param max_iter: max_iter parameter to use in sklearn.cluster.KMeans()
    :param tol: tol parameter to use in sklearn.cluster.KMeans()
    :param random_state: random state to use in sklearn.cluster.KMeans() for reproducible results
    :param coeff: MSSW-specific parameter, 2.66 by default
    :return: True if test_data is drifting, False otherwise
    """
    check_inputs(ref_data, test_data, ref_labels, encoder, scaler)
    if ref_labels is not None:
        ref_labels = pd.DataFrame(LabelEncoder().fit_transform(ref_labels))
    ref_encoded, test_encoded = encode_and_make_numpy(ref_data, test_data, ref_labels, encoder)

    batch_size = test_encoded.shape[0]
    ref_batches = split_to_fixed_size_batches(ref_encoded, batch_size)
    drift_detected = all_drifting_batches(ref_batches, [test_encoded],
                                          n_clusters, n_init, max_iter, tol, random_state,
                                          coeff)[0]
    return drift_detected
