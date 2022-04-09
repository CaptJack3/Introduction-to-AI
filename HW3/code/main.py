import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import RandomState
'''
Currently for debbuging and playing with the functions
'''


def get_dataset_split(train_set: np.array, test_set: np.array,
                      target_attribute: str):
    """
    Splits a dataset into a sample and label set, returning a tuple for each.
    :param train_set: train set
    :param test_set: test set
    :param target_attribute: attribute for classifying, the label column
    :return: A tuple of train and test datasets split.
    """
    # separate target from predictors
    x_train = np.array(train_set.drop(target_attribute, axis=1).copy())
    y_train = np.array(train_set[target_attribute].copy())

    x_test = np.array(test_set.drop(target_attribute, axis=1).copy())
    y_test = np.array(test_set[target_attribute].copy())

    return x_train, y_train, x_test, y_test


def load_data_set(clf_type: str):
    """
    Uses pandas to load train and test dataset.
    :param clf_type: a string equals 'ID3' or 'KNN'
    :return: A tuple of attributes_names (the features row) with train and test datasets split.
    """
    assert clf_type in ('ID3',
                        'KNN'), 'The parameter clf_type must be ID3 or KNN'
    hw_path = str(pathlib.Path(__file__).parent.absolute())
    dataset_path = hw_path + f"/{clf_type}-dataset/"
    train_file_path = dataset_path + "train.csv"
    test_file_path = dataset_path + "test.csv"
    # Import all columns omitting the fist which consists the names of the attributes
    train_dataset = pd.read_csv(train_file_path)
    test_dataset = pd.read_csv(test_file_path)
    attributes_names = list(
        pd.read_csv(train_file_path, delimiter=",", dtype=str, nrows=1).keys())
    return attributes_names, train_dataset, test_dataset


def test_class_counts_id3():
    x_train, y_train, x_test, y_test, attributes_names = load_i3d_data()
    small_x_train = x_train[:5, :]
    small_y_train = y_train[:5]
    train_counts = class_counts(small_x_train,
                                small_y_train)  # returns a dictionary
    B_counts = train_counts["B"]
    M_counts = train_counts["M"]


def load_i3d_data():
    attributes_names, train_dataset, test_dataset = load_data_set("ID3")
    data_split = get_dataset_split(train_dataset, test_dataset, "diagnosis")
    (x_train, y_train, x_test, y_test) = data_split
    return x_train, y_train, x_test, y_test, attributes_names


if __name__ == "__main__":
    test_class_counts_id3()