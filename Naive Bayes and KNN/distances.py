import numpy as np


def euclidean_for_loop(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Add doc string here
    :param x1: Numpy 1D array/vector of any length
    :param x2: Numpy 1D array/vector of any length
    :return: a single double or int depending on the array type
    """

    # this is to initialize a temp variable to add our running total too
    temp = 0.00

    # This code will subtract, square the different between each element between the two arrays. It will then
    # add this number to the running total
    for i in range(x1.size):
        temp += (x2[i] - x1[i]) ** 2
    return temp ** (1 / 2)


def euclidean_vectorized(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Add doc string here
    :param x1: Numpy 1D array/vector of any length
    :param x2: Numpy 1D array/vector of any length
    :return: a single double or int depending on the array type
    """
    # This code will preform the same function as the euclidean for loop above.
    # The only difference is we will use the vectorized implementation build inherently into the python numpy functions.

    return np.sqrt(np.sum(np.square(x1 - x2)))
