import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

import numpy as np
import math
from scipy.stats import norm, gaussian_kde
from sklearn.utils import resample
from statistics import variance


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


# Test statistic

def delta(S2, Sprime, kde):
    return np.sum(kde.logpdf(Sprime.T)) - (len(Sprime) / len(S2)) * np.sum(kde.logpdf(S2.T))


# Determining the critical value

def est_var_estimates(S2, estSize, kde):
    Est = []
    std_prev_t = 0.0
    for t in range(estSize):
        R = resample(S2)
        densities = kde.logpdf(R.T)
        Est.append((len(S2) / (len(S2) - 1)) * variance(densities))

        # stopping criterion after 30 estimates if their SD stabilizes within 1 percent
        std_current_t = np.std(Est)
        diff_std = float('inf')
        if t > 1:
            diff_std = abs(std_prev_t - std_current_t) / std_prev_t
        if t >= 29 and diff_std < 0.01:
            break
        std_prev_t = std_current_t

    Est.sort()
    return Est


def critical_value(p, stepSize, S2_size, Sprime_size, kde, Est):
    M = math.floor(p / stepSize - 1)
    C = []
    for i in range(M):
        alpha = (i + 1) * stepSize
        beta = p - alpha
        # estimate variance for this beta:
        upper_limit = Est[math.ceil((len(Est) * (1 - beta) - 1))]
        var = (Sprime_size + Sprime_size**2/S2_size) * upper_limit
        # find c such that P(D <= c) = alpha, D ~ N(0, std):
        D = norm(0, np.sqrt(var))
        c = D.ppf(alpha)
        C.append(c)
    Cmax = np.amin(C)
    return Cmax


# Full procedure

def density_test(S2, Sprime, p, kde, Est):
    # Calculate delta between S2 and S'
    d = delta(S2, Sprime, kde)
    # Get critical value from S2
    stepSize = 0.002
    c = critical_value(p, stepSize, len(S2), len(Sprime), kde, Est)
    # Report drift if delta < critical value
    return d < c, d, c


def scd_all(raw_ref_data, raw_test_batches, ref_labels=None, encoder=None, scaler=None, p=0.08, bidirectional=False):

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

    # Randomly partition the training set S into S1 and S2
    S1, S2 = train_test_split(ref_data, train_size=0.50, shuffle=True)

    # Learn the kernel model over S1
    kde = gaussian_kde(S1.to_numpy().T)

    # Estimate the variance
    estSize = 4000
    Est = est_var_estimates(S2.to_numpy(), estSize, kde)

    deltas = []
    crits = []
    deltas_reverse = []
    crits_reverse = []
    drifts = []

    # Consider each test batch S':
    for batch in range(len(test_batches)):
        p_value = p
        if bidirectional:
            p_value = p/2

        detected_scd, delta, crit = density_test(
            S2.to_numpy(),
            test_batches[batch].to_numpy(),
            p_value,
            kde,
            Est
        )
        deltas.append(delta)
        deltas_reverse.append(0)
        crits.append(crit)
        crits_reverse.append(0)
        if detected_scd:
            drifts.append(batch + 1)

        # If no drift was detected, run again with S and S' reversed
        elif bidirectional:
            S1_reverse, S2_reverse = train_test_split(
                test_batches[batch], train_size=0.50, shuffle=True)

            # Learn the kernel model over S1_reverse
            kde_reverse = gaussian_kde(
                S1_reverse.to_numpy().T)

            # Estimate the variance
            Est_reverse = est_var_estimates(
                S2_reverse.to_numpy(), estSize, kde_reverse)

            detected_scd_reverse, delta_reverse, crit_reverse = density_test(
                S2_reverse.to_numpy(),
                ref_data.to_numpy(),
                p_value,
                kde_reverse,
                Est_reverse
            )
            deltas_reverse[batch] = delta_reverse
            crits_reverse[batch] = crit_reverse
            if detected_scd_reverse:
                drifts.append(batch + 1)

    return drifts


def scd(ref_data, test_data, ref_labels=None, encoder=None, scaler=None, p=0.08, bidirectional=False):
    drifts = scd_all(ref_data, [test_data], ref_labels,
                     encoder, scaler, p, bidirectional)
    return len(drifts) == 1
