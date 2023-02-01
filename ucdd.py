"""
Drift detection algorithm from
[1] D. Shang, G. Zhang, and J. Lu,
“Fast concept drift detection using unlabeled data,”
in Developments of Artificial Intelligence Technologies in Computation and Robotics,
Cologne, Germany, Oct. 2020, pp. 133–140. doi: 10.1142/9789811223334_0017.

@author: Jindrich POHL

- Unless specified otherwise, functions in this file work with numpy arrays
- The terms "batch" and "window" mean the same thing
"""
import itertools

import numpy as np
import pandas as pd
import sklearn
from scipy.io import arff
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import scipy
from pyclustering.utils import distance_metric, type_metric
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def print_label_cluster_stats(actual_label_cluster, cluster_name):
    """
    Print out how well k-means predicted classes by one cluster

    :param actual_label_cluster: actual labels of points in this cluster
    :param cluster_name: the name of this cluster - "plus" or "minus"
    :return:
    """
    cluster_size = np.shape(actual_label_cluster)[0]
    print('number of points in the', cluster_name, 'cluster:', cluster_size)
    class1_perc = 100 * np.sum(actual_label_cluster) / cluster_size
    print('actual class 1 percentage in the', cluster_name, 'cluster:',
          class1_perc)
    print('actual class 0 percentage in the', cluster_name, 'cluster:',
          100 - class1_perc)


def split_back_to_windows(window_union, labels, len_ref_window, len_test_window, label_batch_union=None):
    """
    Separate predicted points back to original reference and testing windows (through boolean masks)

    :param window_union: array of shape (len_ref_window + len_test_window, #attributes)
    :param labels: list of cluster predictions of points in window_union
    :param len_ref_window: #points in the reference window
    :param len_test_window: #points in the testing window
    :return: 2d arrays of X0+, X0-, X1+, X1-, and the k-means class estimate accuracies
    """
    ref_mask = np.concatenate([np.repeat(True, len_ref_window), np.repeat(False, len_test_window)])
    plus_mask = np.where(labels == 1, True, False)

    ref_plus_mask = np.logical_and(ref_mask, plus_mask)
    ref_minus_mask = np.logical_and(ref_mask, np.logical_not(plus_mask))
    test_plus_mask = np.logical_and(np.logical_not(ref_mask), plus_mask)
    test_minus_mask = np.logical_and(np.logical_not(ref_mask), np.logical_not(plus_mask))

    cluster_classif_acc = None

    if label_batch_union is not None:
        total_points = len_ref_window + len_test_window
        print('label batch union shape')
        print(label_batch_union.shape)
        print('plus mask shape')
        print(plus_mask.shape)
        num_class1_in_plus = np.count_nonzero(label_batch_union[plus_mask])
        num_class0_in_minus = np.count_nonzero(label_batch_union[~plus_mask] == 0)
        perc_correctly_classified_points_v1 = (num_class1_in_plus + num_class0_in_minus) / total_points

        num_class0_in_plus = np.count_nonzero(label_batch_union[plus_mask] == 0)
        num_class1_in_minus = np.count_nonzero(label_batch_union[~plus_mask])
        perc_correctly_classified_points_v2 = (num_class0_in_plus + num_class1_in_minus) / total_points

        cluster_classif_acc = max(perc_correctly_classified_points_v1,
                                  perc_correctly_classified_points_v2)

    ref_plus = window_union[ref_plus_mask]
    ref_minus = window_union[ref_minus_mask]
    test_plus = window_union[test_plus_mask]
    test_minus = window_union[test_minus_mask]

    return ref_plus, ref_minus, test_plus, test_minus, cluster_classif_acc


def join_predict_split(ref_window, test_window,
                       n_init, max_iter, tol, random_state,
                       reference_label_batch=None,
                       testing_label_batch=None):
    """
    Join points from two windows, predict their labels through kmeans, then separate them again

    :param ref_window: array of shape (#points in ref_window, #attributes)
    :param test_window: array of shape (#points in test_window, #attributes)
    :param n_init: see sklearn.cluster.KMeans n_init
    :param max_iter: see sklearn.cluster.KMeans max_iter
    :param tol: see sklearn.cluster.KMeans tol
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :return: 2d arrays of X0+, X0-, X1+, X1-, and the k-means class estimate accuracies
    """
    # join the points from two windows
    window_union = np.vstack((ref_window, test_window))

    # predict their label values
    predicted_labels = KMeans(n_clusters=2, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)\
        .fit_predict(window_union)

    label_batch_union = None
    if reference_label_batch is not None and testing_label_batch is not None:
        label_batch_union = np.vstack((reference_label_batch, testing_label_batch))

    # split values by predicted label and window
    return split_back_to_windows(window_union, predicted_labels, ref_window.shape[0], test_window.shape[0],
                                 label_batch_union)


def compute_neighbors(u, v, debug_string='v'):
    """
    Find the indices of nearest neighbors of v in u

    :param u: array of shape (#points in u, #attributes),
    :param v: array of shape (#points in v, #attributes),
    :param debug_string: string to use in debug print statements
    :return: array of shape (#unique nearest neighbour indices of v in u, 1)
    """
    neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    neigh.fit(v)

    neigh_ind_v = neigh.kneighbors(u, return_distance=False)
    unique_v_neighbor_indices = np.unique(neigh_ind_v)
    w = v[unique_v_neighbor_indices]
    return w


def compute_beta(u, v0, v1, beta_x=0.5, debug=False):
    """
    Find neighbors and compute beta based on u, v0, v1

    :param u: array of shape (#points in u, #attributes), cluster U from the algorithm
    :param v0: array of shape (#points in v0, #attributes), cluster V0 from the algorithm
    :param v1: array of shape (#points in v1, #attributes), cluster V1 from the algorithm
    :param beta_x: default=0.5, x to use in the Beta distribution
    :param debug: debug: flag for helpful print statements
    :return: (beta - the regular beta cdf value,
        beta_additional - the beta cdf value for exchanged numbers of neighbours as parameters)
    """
    # if there is so much imbalance that at least one cluster is empty, report drift immediately
    if min(len(u), len(v0), len(v1)) == 0:
        beta = 0
        beta_additional = 0
    else:
        w0 = compute_neighbors(u, v0, 'v0')
        w1 = compute_neighbors(u, v1, 'v1')
        if debug: print('neighbors in W0', len(w0))
        if debug: print('neighbors in W1', len(w1))
        beta = scipy.stats.beta.cdf(beta_x, len(w0), len(w1))
        beta_additional = scipy.stats.beta.cdf(beta_x, len(w1), len(w0))
        if debug: print('beta', beta)
        if debug: print('beta additional', beta_additional)
    return beta, beta_additional


def concept_drift_detected(
        ref_window,
        test_window,
        additional_check,
        n_init,
        max_iter,
        tol,
        random_state,
        threshold=0.05,
        debug=False,
        reference_label_batch=None,
        testing_label_batch=None
):
    """
    Detect whether a concept drift occurred based on one reference and one testing window of same sizes

    :param ref_window: array of shape (#points in this reference window, #attributes)
    :param test_window: array of shape (#points in this testing window, #attributes)
    :param additional_check: whether to use a two-fold test or not
    :param n_init: see sklearn.cluster.KMeans n_init
    :param max_iter: see sklearn.cluster.KMeans max_iter
    :param tol: see sklearn.cluster.KMeans tol
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :param threshold: default=0.05, statistical threshold to detect drift
    :param debug: flag for helpful print statements
    :return: (true if drift is detected based on the two windows and false otherwise,
    k-means class estimate accuracies)

    """
    ref_plus, ref_minus, test_plus, test_minus, cluster_classif_acc = \
        join_predict_split(ref_window, test_window,
                           n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state,
                           reference_label_batch=reference_label_batch, testing_label_batch=testing_label_batch)

    if debug: print('BETA MINUS (ref+, ref-, test-)')
    beta_minus, beta_minus_additional = compute_beta(
        ref_plus, ref_minus, test_minus, debug=debug)
    if debug: print('BETA PLUS (ref-, ref+, test+)')
    beta_plus, beta_plus_additional = compute_beta(
        ref_minus, ref_plus, test_plus, debug=debug)

    drift = (beta_plus < threshold or beta_minus < threshold)
    if additional_check:
        drift = drift | (beta_plus_additional < threshold or beta_minus_additional < threshold)

    return drift, cluster_classif_acc


def all_drifting_batches(
        reference_data_batches,
        testing_data_batches,
        min_ref_batches_drift,
        additional_check,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        parallel=True,
        reference_label_batches=None,
        testing_label_batches=None,
        debug=False
):
    """
    Find all drift locations based on the given reference and testing batches

    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :param min_ref_batches_drift: the minimum fraction of reference batches that must signal drift for one test batch
    :param additional_check: whether to use a two-fold test or not
    :param n_init: default=10, see sklearn.cluster.KMeans n_init
    :param max_iter: default=300, see sklearn.cluster.KMeans max_iter
    :param tol: default=1e-4, see sklearn.cluster.KMeans tol
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :param parallel: whether drift detections should happen on all cores - warning, very computationally heavy!
    :param reference_label_batches: used for class estimate accuracy, only applicable if parallel=False,
        list of arrays of shape (n_r_r, 1), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_label_batches: used for class estimate accuracy, only applicable if parallel=False,
        list of arrays of shape (n_r_t, 1), r_t=testing batch number,
        n_r_t=#points in this batch
    :return: a boolean list, length=len(testing_data_batches),
        an entry is True if drift was detected there and False otherwise
    """

    if parallel:
        drifts_detected = all_drifting_batches_parallel(
            reference_data_batches,
            testing_data_batches,
            min_ref_batches_drift,
            additional_check,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )
    else:
        drifts_detected = []
        for i, test_window in enumerate(testing_data_batches):
            if testing_label_batches is not None:
                current_test_label_batch = testing_label_batches[i]
            num_ref_drifts = 0 # how many training batches signal drift against this testing batch
            for j, ref_window in enumerate(reference_data_batches):
                if reference_label_batches is not None:
                    current_ref_label_batch = reference_label_batches[j]
                    drift_here, cluster_predict_acc = concept_drift_detected(
                        ref_window, test_window, additional_check, n_init, max_iter, tol, random_state,
                        reference_label_batch=reference_label_batches[j],
                        testing_label_batch=testing_label_batches[i],
                        debug=debug
                    )
                else:
                    drift_here, cluster_predict_acc = concept_drift_detected(
                        ref_window, test_window, additional_check, n_init, max_iter, tol, random_state,
                        debug=debug
                    )
                if drift_here:
                    num_ref_drifts += 1

            drift = (num_ref_drifts / len(reference_data_batches)) > min_ref_batches_drift
            drifts_detected.append(drift)

    return drifts_detected


def get_final_drifts_from_all_info(drifts_2d_arr, len_ref_data_batches, min_ref_batches_drift):
    """
    Convert outputs of all_drifting_batches_parallel_all_info to a list of drift detections

    :param drifts_2d_arr: array of shape (len_ref_data_batches, #testing batches) with drift detection results for each
    batch combination
    :param len_ref_data_batches: #reference batches
    :param min_ref_batches_drift: minimum fraction of reference batches detected as drifting for the corresponding
    testing batch to be drifting
    :return: a boolean list, length=#testing batches,
        an entry is True if drift was detected there and False otherwise
    """
    num_signals_each_testing_batch = np.sum(drifts_2d_arr, axis=0)
    drifts_detected = ((num_signals_each_testing_batch / len_ref_data_batches) > min_ref_batches_drift).tolist()
    return drifts_detected


def all_drifting_batches_parallel(
        reference_data_batches,
        testing_data_batches,
        min_ref_batches_drift,
        additional_check,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        reference_label_batches=None,
        testing_label_batches=None
):
    """
    Find all drift locations based on the given reference and testing batches in parallel

    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :param min_ref_batches_drift: the minimum fraction of reference batches that must signal drift for one test batch
    :param additional_check: whether to use a two-fold test or not
    :param n_init: default=10, see sklearn.cluster.KMeans n_init
    :param max_iter: default=300, see sklearn.cluster.KMeans max_iter
    :param tol: default=1e-4, see sklearn.cluster.KMeans tol
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :param reference_label_batches: used for class estimate accuracy,
        list of arrays of shape (n_r_r, 1), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_label_batches: used for class estimate accuracy,
        list of arrays of shape (n_r_t, 1), r_t=testing batch number,
        n_r_t=#points in this batch
    :return: a boolean list, length=len(testing_data_batches),
        an entry is True if drift was detected there and False otherwise
    """
    drifts_2d_arr, cluster_classif_accs_2d_arr = all_drifting_batches_parallel_all_info(
        reference_data_batches,
        testing_data_batches,
        additional_check,
        n_init,
        max_iter,
        tol,
        random_state,
        reference_label_batches,
        testing_label_batches
    )
    drifts_detected = get_final_drifts_from_all_info(drifts_2d_arr, len(reference_data_batches), min_ref_batches_drift)
    return drifts_detected


def all_drifting_batches_parallel_all_info(
        reference_data_batches,
        testing_data_batches,
        additional_check,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=None,
        reference_label_batches=None,
        testing_label_batches=None
):
    """
    Find all drift locations and all classification accuracies for each combination of reference and testing batches,
    in parallel

    :param reference_data_batches: list of arrays of shape (n_r_r, #attributes), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_data_batches: list of arrays of shape (n_r_t, #attributes), r_t=testing batch number,
        n_r_t=#points in this batch
    :param additional_check: whether to use a two-fold test or not
    :param n_init: default=10, see sklearn.cluster.KMeans n_init
    :param max_iter: default=300, see sklearn.cluster.KMeans max_iter
    :param tol: default=1e-4, see sklearn.cluster.KMeans tol
    :param random_state: used to potentially control randomness - see sklearn.cluster.KMeans random_state
    :param reference_label_batches: used for class estimate accuracy,
        list of arrays of shape (n_r_r, 1), r_r=reference batch number,
        n_r_r=#points in this batch
    :param testing_label_batches: used for class estimate accuracy,
        list of arrays of shape (n_r_t, 1), r_t=testing batch number,
        n_r_t=#points in this batch
    :return: (drifts_2d_arr array of shape (len(reference_data_batches), len(testing_data_batches)) with boolean values
    indicating for which combination of reference and testing batch a drift was detected,
    cluster_classif_accs_2d_arr array of shape (len(reference_data_batches), len(testing_data_batches) with float
    accuracies of class estimates by k-means clustering)
    """
    threshold = 0.05
    debug = False

    pool_iterables = []

    for i, test_window in enumerate(testing_data_batches):
        testing_label_batch = None
        if testing_label_batches is not None:
            testing_label_batch = testing_label_batches[i]
        for j, ref_window in enumerate(reference_data_batches):
            reference_label_batch = None
            if reference_label_batches is not None:
                reference_label_batch = reference_label_batches[j]
            entry = (ref_window, test_window,
                     additional_check,
                     n_init, max_iter, tol,
                     random_state,
                     threshold,
                     debug,
                     reference_label_batch,
                     testing_label_batch)
            pool_iterables.append(entry)

    with Pool() as pool:
        drifts_and_cluster_classif_acc_1d = pool.starmap(concept_drift_detected, pool_iterables)

    drifts_1d_tuple, cluster_classif_accs_1d_tuple = tuple(zip(*drifts_and_cluster_classif_acc_1d))
    drifts_1d_arr = np.asarray(drifts_1d_tuple)
    cluster_classif_accs_1d_arr = np.asarray(cluster_classif_accs_1d_tuple)

    drifts_2d_arr = drifts_1d_arr.reshape((len(testing_data_batches), len(reference_data_batches))).T
    cluster_classif_accs_2d_arr = cluster_classif_accs_1d_arr.reshape(
        (len(testing_data_batches), len(reference_data_batches))).T

    return drifts_2d_arr, cluster_classif_accs_2d_arr








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
        raise ValueError('UCDD uses k-means for which scaling is highly desired, so scaler cannot be None')
    if scaler != 'minmax':
        raise ValueError('Only minmax scaling is supported, so the given scaler: ' + scaler.__str__()
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


def ucdd(ref_data, test_data, ref_labels=None, encoder=None, scaler='minmax',
         n_init=100, max_iter=1000, tol=0, random_state=None,
         min_ref_batches_drift=0.3, additional_check=True,
         parallel=False):
    """
    Determine whether test_data is drifting by comparing it to ref_data

    :param ref_data: dataframe of shape (#reference points, #features)
    :param test_data: dataframe of shape (#testing points, #features)
    :param ref_labels: optional dataframe with class labels of shape (#reference points, 1)
    :param encoder: None if categorical variables should be excluded, 'onehot' or 'target' if you wish to encode them
    :param scaler: always 'minmax', only present here for API consistency
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

    # minmax scaling done by default
    scaler = MinMaxScaler()
    scaler.fit(ref_encoded)
    ref_encoded = scaler.transform(ref_encoded)
    test_encoded = scaler.transform(test_encoded)

    batch_size = test_encoded.shape[0]
    ref_batches = split_to_fixed_size_batches(ref_encoded, batch_size)

    drift_detected = all_drifting_batches(ref_batches, [test_encoded],
                                          min_ref_batches_drift, additional_check,
                                          n_init, max_iter, tol, random_state,
                                          parallel)[0]
    return drift_detected
