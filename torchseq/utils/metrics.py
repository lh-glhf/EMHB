import os.path
import sys
import numpy as np
from scipy.signal import find_peaks
# from dtw_metric import accelerated_dtw
import time
from fastdtw import fastdtw

"""
 def metric(pred, true):
    return mae, mse, rmse, mape, mspe, smape, wape, msmape
"""


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def SMAPE(pred, true):
    denominator = (np.abs(true) + np.abs(pred)) / 2
    return np.mean(np.abs(pred - true) / denominator)


def WAPE(pred, true):
    return np.sum(np.abs(pred - true)) / np.sum(np.abs(true))


def MSMAPE(pred, true, epsilon=0.1):
    denominator = np.maximum((np.abs(true) + np.abs(pred) + epsilon), 0.5 + epsilon) / 2
    return np.mean(np.abs(pred - true) / denominator)


def EuclideanDistance(pred, true, concat=False):
    # b, l ,d
    if concat:
        return np.sum(np.sum(np.sqrt(np.sum((pred - true) ** 2, axis=-1)), axis=-1))
    else:
        return np.mean(np.sum(np.sqrt(np.sum((pred - true) ** 2, axis=-1)), axis=-1))


def DTW(pred, true):
    dtw_list = []
    manhattan_distance = lambda x, y: np.abs(x - y)
    for i in range(pred.shape[0]):
        x = pred[i].reshape(-1, 1)
        y = true[i].reshape(-1, 1)
        if i % 5 == 0:
            print("calculating dtw iter:", i)
        d, _ = fastdtw(x, y, dist=manhattan_distance)
        dtw_list.append(d)
    dtw = np.array(dtw_list).mean()
    return dtw


def _compute_derivative(time_series):
    """
    计算时间序列的一阶导数
    """
    derivative = np.zeros_like(time_series)
    for t in range(1, len(time_series) - 1):
        derivative[t] = ((time_series[t] - time_series[t - 1]) + ((time_series[t + 1] - time_series[t - 1]) / 2)) / 2
    derivative[0] = derivative[1]
    derivative[-1] = derivative[-2]
    return derivative


def DDTW(pred, true):
    ddtw_list = []
    manhattan_distance = lambda x, y: np.abs(x - y)
    for i in range(pred.shape[0]):
        x = pred[i]
        y = true[i]

        dx = _compute_derivative(x)
        dy = _compute_derivative(y)

        dx = dx.reshape(-1, 1)
        dy = dy.reshape(-1, 1)
        if i % 5 == 0:
            print("calculating ddtw iter:", i)

        d, _ = fastdtw(dx, dy, dist=manhattan_distance)
        ddtw_list.append(d)
    ddtw = np.array(ddtw_list).mean()
    return ddtw


def _count_peaks(vector):
    peaks, _ = find_peaks(vector)
    return len(peaks)


def changePoints(pred, true, concat=False):  # b, l, d
    if not concat:
        pred = np.transpose(pred, (0, 2, 1))
        true = np.transpose(true, (0, 2, 1))
        pred_counts = []
        for b in range(pred.shape[0]):
            sum_dim_count = 0
            for d in range(pred.shape[1]):
                vector = pred[b, d, :]
                sum_dim_count += _count_peaks(vector)
            pred_counts.append(sum_dim_count)
        true_counts = []
        for b in range(true.shape[0]):
            sum_dim_count = 0
            for d in range(true.shape[1]):
                vector = true[b, d, :]
                sum_dim_count += _count_peaks(vector)
            true_counts.append(sum_dim_count)
        pred_counts = np.array(pred_counts)
        true_counts = np.array(true_counts)
        return np.mean(np.abs(true_counts - pred_counts) / true_counts)
    else:
        pred = np.transpose(pred.reshape(-1, pred.shape[2]), (1, 0))
        true = np.transpose(true.reshape(-1, true.shape[2]), (1, 0))
        pred_dim_count = 0
        for d in range(pred.shape[0]):
            vector = pred[d, :]
            pred_dim_count += _count_peaks(vector)
        true_dim_count = 0
        for d in range(true.shape[0]):
            vector = true[d, :]
            true_dim_count += _count_peaks(vector)
        return np.abs(true_dim_count - pred_dim_count) / true_dim_count


def _calculate_entropy(vector):
    _, counts = np.unique(vector, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def _calculate_matrix_entropy(matrix, concat=False):
    if concat:
        reshaped_matrix = matrix.reshape(-1, matrix.shape[2])
        entropy_per_feature = np.zeros(matrix.shape[2])
        for i in range(matrix.shape[2]):
            entropy_per_feature[i] = _calculate_entropy(reshaped_matrix[:, i])
        return np.mean(entropy_per_feature)
    else:
        reshaped_matrix = np.transpose(matrix, (0, 2, 1))  # b, d, l
        entropy_per_sample = np.zeros((reshaped_matrix.shape[0], reshaped_matrix.shape[1]))
        for i in range(reshaped_matrix.shape[0]):
            for j in range(reshaped_matrix.shape[1]):
                entropy_per_sample[i, j] = _calculate_entropy(reshaped_matrix[i, j, :])
        return np.mean(entropy_per_sample)


def entropy_diff(pred, true, concat=False):
    entropy_pred = _calculate_matrix_entropy(pred, concat)
    entropy_true = _calculate_matrix_entropy(true, concat)
    return np.abs(entropy_pred - entropy_true) / entropy_true


def _calculate_matrix_variance(matrix, concat=False):
    if concat:
        reshaped_matrix = matrix.reshape(-1, matrix.shape[2])
        variance_per_feature = np.zeros(matrix.shape[2])
        for i in range(matrix.shape[2]):
            variance_per_feature[i] = np.var(reshaped_matrix[:, i])
        return np.mean(variance_per_feature)
    else:
        reshaped_matrix = np.transpose(matrix, (0, 2, 1))  # b, d, l
        variance_per_sample = np.zeros((reshaped_matrix.shape[0], reshaped_matrix.shape[1]))
        for i in range(reshaped_matrix.shape[0]):
            for j in range(reshaped_matrix.shape[1]):
                variance_per_sample[i, j] = np.var(reshaped_matrix[i, j, :])
        return np.mean(variance_per_sample)


def stas(pred, true, concat=False):
    avg_diff = np.abs(np.mean(pred) - np.mean(true)) / np.mean(true)
    var_pred = _calculate_matrix_variance(pred, concat)
    var_true = _calculate_matrix_variance(true, concat)
    var_diff = np.abs(var_pred - var_true) / var_true
    return avg_diff, var_diff


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    wape = WAPE(pred, true)
    msmape = MSMAPE(pred=pred, true=true)
    return mae, mse, rmse, mape, mspe, smape, wape, msmape


def metric_all(pred, true, concat=False):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    wape = WAPE(pred, true)
    msmape = MSMAPE(pred=pred, true=true)
    ed = EuclideanDistance(pred, true, concat)
    if concat:
        dtw = DTW(pred, true)
        ddtw = DDTW(pred, true)
    else:
        dtw = 0
        ddtw = 0
    peaks = changePoints(pred, true, concat)
    entropy = entropy_diff(pred, true, concat)
    avg, var = stas(pred, true, concat)
    return mae, mse, rmse, mape, mspe, smape, wape, msmape, ed, dtw, ddtw, peaks, entropy, avg, var


def calFileTest(rootPath):
    pred_path = os.path.join(rootPath, "pred.npy")
    true_path = os.path.join(rootPath, "true.npy")
    pred = np.load(pred_path)
    true = np.load(true_path)
    print("Rolling Test shape:", pred.shape)
    B, L, D = pred.shape
    """
        Generate noRolling result
    """
    indices = list(range(0, B, L))
    predWithoutRoll = pred[indices, :, :]
    trueWithoutRoll = true[indices, :, :]
    print("noRolling Test shape:", predWithoutRoll.shape)

    print("Rolling Test Result:")
    mae, mse, rmse, mape, mspe, smape, wape, msmape, ed, dtw, ddtw, peaks, entropy, avg, var = metric_all(pred, true,
                                                                                                          concat=False)
    print("Metrics:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    print(f"Mean Squared Percentage Error (MSPE): {mspe}")
    print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape}")
    print(f"Weighted Absolute Percentage Error (WAPE): {wape}")
    print(f"Mean Scaled Mean Absolute Percentage Error (MSMAPE): {msmape}")
    print(f"Euclidean Distance (ED): {ed}")
    print(f"Dynamic Time Warping (DTW): {dtw}")
    print(f"Derivative Dynamic Time Warping (DDTW): {ddtw}")
    print(f"Number of Peaks : {peaks}")
    print(f"Entropy: {entropy}")
    print(f"Average: {avg}")
    print(f"Variance: {var}")

    print("noRolling Test Result:")
    mae, mse, rmse, mape, mspe, smape, wape, msmape, ed, dtw, ddtw, peaks, entropy, avg, var = metric_all(
        predWithoutRoll,
        trueWithoutRoll,
        concat=False)
    print("Metrics:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    print(f"Mean Squared Percentage Error (MSPE): {mspe}")
    print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape}")
    print(f"Weighted Absolute Percentage Error (WAPE): {wape}")
    print(f"Mean Scaled Mean Absolute Percentage Error (MSMAPE): {msmape}")
    print(f"Euclidean Distance (ED): {ed}")
    print(f"Dynamic Time Warping (DTW): {dtw}")
    print(f"Derivative Dynamic Time Warping (DDTW): {ddtw}")
    print(f"Number of Peaks: {peaks}")
    print(f"Entropy: {entropy}")
    print(f"Average: {avg}")
    print(f"Variance: {var}")


if __name__ == "__main__":
    rootPath = sys.argv[1]
    calFileTest(rootPath)
