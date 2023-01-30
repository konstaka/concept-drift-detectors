import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import rankdata, norm


# Preprocessing

def ensure_df(data):
    checked_data = data
    if not isinstance(data, pd.DataFrame):
        checked_data = pd.DataFrame(data)
    return checked_data


def encode(ref_data, test_batches, ref_labels, encoder=None):
    encoded_ref_data = ref_data
    encoded_test_batches = test_batches

    if encoder == 'onehot':
        # OneHotEncoder
        ohe = OneHotEncoder()
        ref_data_num = ref_data.select_dtypes(include=[np.number])
        ref_data_cat = ref_data.select_dtypes(exclude=[np.number])
        encoded_ref_data_cat = ohe.fit_transform(ref_data_cat).toarray()
        encoded_ref_data = pd.DataFrame(np.concatenate(
            [ref_data_num, encoded_ref_data_cat], axis=1))
        encoded_test_batches = []
        for batch in range(len(test_batches)):
            test_data_num = test_batches[batch].select_dtypes(include=[
                                                              np.number])
            test_data_cat = test_batches[batch].select_dtypes(exclude=[
                                                              np.number])
            encoded_test_data_cat = ohe.transform(test_data_cat).toarray()
            encoded_test_batches.append(pd.DataFrame(np.concatenate(
                [test_data_num, encoded_test_data_cat], axis=1)))
    elif encoder == 'ordinal':
        # OrdinalEncoder
        oe = OrdinalEncoder()
        ref_data_num = ref_data.select_dtypes(include=[np.number])
        ref_data_cat = ref_data.select_dtypes(exclude=[np.number])
        encoded_ref_data_cat = oe.fit_transform(ref_data_cat)
        encoded_ref_data = pd.DataFrame(np.concatenate(
            [ref_data_num, encoded_ref_data_cat], axis=1))
        encoded_test_batches = []
        for batch in range(len(test_batches)):
            test_data_num = test_batches[batch].select_dtypes(include=[
                                                              np.number])
            test_data_cat = test_batches[batch].select_dtypes(exclude=[
                                                              np.number])
            encoded_test_data_cat = oe.transform(test_data_cat)
            encoded_test_batches.append(pd.DataFrame(np.concatenate(
                [test_data_num, encoded_test_data_cat], axis=1)))
    elif encoder == 'target' and ref_labels is not None:
        # TargetEncoder
        cols = ref_data.select_dtypes(exclude=[np.number]).columns
        te = TargetEncoder(cols, smoothing=0, return_df=True)
        encoded_ref_data = te.fit_transform(ref_data, ref_labels)
        encoded_test_batches = []
        for batch in range(len(test_batches)):
            encoded_test_batches.append(te.transform(test_batches[batch]))

    return encoded_ref_data, encoded_test_batches


def scale(ref_data, test_batches, scaler=None):
    scaled_ref_data = ref_data
    scaled_test_batches = test_batches

    if scaler == 'minmax':
        mms = MinMaxScaler()
        scaled_ref_data = pd.DataFrame(mms.fit_transform(ref_data))
        scaled_test_batches = list(
            map(lambda batch: pd.DataFrame(mms.transform(batch)), test_batches))

    return scaled_ref_data, scaled_test_batches


# SyncStream-PCA

def syncstream_pca_all(raw_ref_data, raw_test_batches, ref_labels=None, encoder=None, scaler=None, consecutive=False):

    # Make sure data is in DataFrames
    ref_data = ensure_df(raw_ref_data)
    test_batches = []
    for batch in range(len(raw_test_batches)):
        test_batches.append(ensure_df(raw_test_batches[batch]))

    # Encode and scale the data
    if encoder:
        ref_data, test_batches = encode(
            ref_data, test_batches, ref_labels, encoder)
    if scaler:
        ref_data, test_batches = scale(ref_data, test_batches, scaler)

    deltas = []
    crits = []
    drifts = []

    pca = PCA(n_components=1)

    pca.fit(ref_data.values)
    ref_eigenvector = pca.components_[0]

    for batch in range(len(test_batches)):
        pca.fit(test_batches[batch].values)
        batch_eigenvector = pca.components_[0]
        measured_angle = np.degrees(
            np.arccos(np.dot(batch_eigenvector, ref_eigenvector)))
        crit = 30

        if (consecutive):
            ref_eigenvector = batch_eigenvector

        deltas.append(measured_angle)
        crits.append(crit)
        if (measured_angle > crit):
            drifts.append(batch + 1)

    return drifts


def syncstream_pca(ref_data, test_data, ref_labels=None, encoder=None, scaler=None):
    drifts = syncstream_pca_all(
        ref_data, [test_data], ref_labels, encoder, scaler)
    return len(drifts) == 1


# SyncStream-Stat

def midrank(Dt, ranks_in_union, start_i, j):
    rank_sum = 0
    for i in range(start_i, start_i + len(Dt)):
        rank_sum += ranks_in_union[i]
    return rank_sum / len(Dt)


def v2(Dt, u, start_i):
    rank_diff_sum = 0
    for j in range(Dt.shape[1]):
        ranks_in_union = rankdata(u[:, j])
        ranks_in_Dt = rankdata(Dt[:, j])
        midrank_j = midrank(Dt, ranks_in_union, start_i, j)
        for i in range(0, len(Dt)):
            rank_diff_sum += (ranks_in_union[start_i + i] -
                              ranks_in_Dt[i] - midrank_j + (len(Dt) + 1) / 2) ** 2
    return (1 / (len(Dt) - 1)) * rank_diff_sum


def midrank_diff(Dt, Dt1, u, j):
    ranks_in_union = rankdata(u[:, j])
    return midrank(Dt, ranks_in_union, 0, j) - midrank(Dt1, ranks_in_union, len(Dt), j)


def wilcoxon_test(Dt, Dt1, p):
    u = np.concatenate((Dt, Dt1))
    v2_Dt = v2(Dt, u, 0)
    v2_Dt1 = v2(Dt1, u, len(Dt))
    var_BF = (len(Dt) + len(Dt1)) * v2_Dt / len(Dt1) + \
        (len(Dt) + len(Dt1)) * v2_Dt1 / len(Dt)
    sd_BF = np.sqrt(var_BF)
    midrank_diff_sum = 0
    for j in range(Dt.shape[1]):
        midrank_diff_sum += midrank_diff(Dt, Dt1, u, j)
    WBF = np.sqrt(len(Dt) * len(Dt1) / (len(Dt) + len(Dt1))) * \
        midrank_diff_sum / sd_BF
    crit = norm.ppf(1-p/2)
    return abs(WBF) > crit, abs(WBF), crit


def syncstream_stat_all(raw_ref_data, raw_test_batches, ref_labels=None, encoder=None, scaler=None, p=0.01, consecutive=False):

    # Make sure data is in DataFrames
    ref_data = ensure_df(raw_ref_data)
    test_batches = []
    for batch in range(len(raw_test_batches)):
        test_batches.append(ensure_df(raw_test_batches[batch]))

    # Encode and scale the data
    if encoder:
        ref_data, test_batches = encode(
            ref_data, test_batches, ref_labels, encoder)
    if scaler:
        ref_data, test_batches = scale(ref_data, test_batches, scaler)

    deltas = []
    crits = []
    drifts = []

    for batch in range(len(test_batches)):
        ref_batch = ref_data
        if consecutive and batch != 0:
            ref_batch = test_batches[batch - 1]
        detected_stat, delta, crit = wilcoxon_test(
            ref_batch.values, test_batches[batch].values, p)
        deltas.append(delta)
        crits.append(crit)
        if (detected_stat):
            drifts.append(batch + 1)

    return drifts


def syncstream_stat(ref_data, test_data, ref_labels=None, encoder=None, scaler=None, p=0.01):
    drifts = syncstream_stat_all(
        ref_data, [test_data], ref_labels, encoder, scaler, p)
    return len(drifts) == 1
