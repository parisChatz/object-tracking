import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from sklearn.metrics import mean_squared_error


def absolute_error(pred_value, real_value, pred_value2, real_value2, show_plot=False):
    # Tracking evaluation
    error = np.sqrt(np.power(np.subtract(np.transpose(pred_value), real_value), 2) + np.power(
        np.subtract(np.transpose(pred_value2), real_value2), 2))

    if show_plot:
        sns.distplot(error)
        plt.show()

    rms = np.sqrt(np.mean(np.sqrt(error)))
    print(rms)
