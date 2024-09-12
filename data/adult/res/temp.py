import numpy as np

def count_outliers(data, threshold=3, axis=1):
    """
    Count the number of outliers in each row of a dataset using z-score method.

    Parameters:
    data (numpy.ndarray): Input data array.
    threshold (float): Threshold for identifying outliers in terms of z-score. Default is 3.
    axis (int): Axis along which to compute z-scores. Default is 1 (rows).

    Returns:
    numpy.ndarray: Array containing the number of outliers for each row.
    """
    mean = np.mean(data, axis=axis)
    std_dev = np.std(data, axis=axis)
    z_scores = np.abs((data - mean[:, np.newaxis]) / std_dev[:, np.newaxis])
    outliers_count = np.sum(z_scores > threshold, axis=axis)
    return outliers_count

# Example usage:
data = np.array([[1, 2, 1000],
                 [4, 5, 6],
                 [7, 8, 9]])  # Example dataset with multiple rows
outliers_count_per_row = count_outliers(data)
print("Number of outliers per row:")
print(outliers_count_per_row)