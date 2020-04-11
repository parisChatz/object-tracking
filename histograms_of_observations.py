import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math


def obs_analysis(a, x):
    mean_a = np.mean(a - x)
    var_a = np.var(a - x)
    std_a = np.std(a-x)
    print("mean: ",mean_a,"var: ", var_a)

    x = np.linspace(mean_a - 3 * var_a, mean_a + 3 * var_a, 100)
    plt.plot(x, stats.norm.pdf(x, mean_a, var_a))
    plt.show()
