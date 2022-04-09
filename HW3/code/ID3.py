import math

import numpy as np

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *


class ID3:

    def __init__(self,
                 label_names: list,
                 min_for_pruning=0,
                 target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        counts = class_counts(rows, labels)
        impurity = 0.0

        n_labels = len(counts)
        if n_labels == 1.0:  # all eamples have the same label
            return 0.0
        for label in counts:
            prob_label = counts[label] / len(labels)
            impurity -= prob_label * math.log(prob_label, n_labels)

        return impurity

    def info_gain(self, left, left_labels, right, right_labels,
                  current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node,
        minus the weighted impurity of two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0

        children_entropy = (len(left) * self.entropy(left, left_labels) + \
                            len(right) * self.entropy(right, right_labels)) / \
                            (len(left)+len(right))
        info_gain_value = current_uncertainty - children_entropy

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        gain = 0
        false_rows = np.empty((0, len(rows[0])))
        false_labels = np.empty((0))
        true_rows = np.empty((0, len(rows[0])))
        true_labels = np.empty((0))

        assert len(rows) == len(
            labels), 'Rows size should be equal to labels size.'

        for row_idx, row in enumerate(rows):
            if question.match(row):
                true_rows = np.append(true_rows, [row], axis=0)
                true_labels = np.append(true_labels, labels[row_idx])
            else:
                false_rows = np.append(false_rows, [row], axis=0)
                false_labels = np.append(false_labels, labels[row_idx])

        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels,
                              current_uncertainty)

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        best_gain = -np.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # iterate over each feature (column from 0 to 29, given 30 features)
        for column in range(rows.shape[1]):
            feature_values = rows[:, column]
            values_sorted = feature_values.copy()
            values_sorted.sort()
            values_differences = np.diff(values_sorted)
            threshold_list = values_sorted[1:] - 0.5 * values_differences
            for threshold in threshold_list:  # iterate over every possible threshold
                question = Question(self.label_names[column], column, threshold)
                gain, true_rows, true_labels, false_rows, false_labels = \
                    self.partition(rows, labels, question, current_uncertainty)
                if gain > best_gain:  # if we have found the best split so far
                    best_gain = gain
                    best_question = question
                    best_false_rows, best_false_labels = false_rows, false_labels
                    best_true_rows, best_true_labels = true_rows, true_labels

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        best_question = None
        true_branch, false_branch = None, None

        if self.is_leaf(rows, labels):
            return Leaf(rows, labels)
        # else:
        (_, best_question, true_rows, true_labels, false_rows,
         false_labels) = self.find_best_split(rows, labels)
        true_branch = self.build_tree(true_rows, true_labels)
        false_branch = self.build_tree(false_rows, false_labels)
        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        root_decision_node = self.build_tree(x_train, y_train)
        self.tree_root = root_decision_node

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        if node is None:
            node = self.tree_root
        prediction = None

        if isinstance(node, Leaf):
            return node.majority
        else:
            question = node.question
            answer = question.match(row)
            if answer:
                return self.predict_sample(row, node.true_branch)
            return self.predict_sample(row, node.false_branch)

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        y_pred = np.empty((0))

        for row in rows:
            label = self.predict_sample(row, None)
            y_pred = np.append(y_pred, label)

        return y_pred

    def is_leaf(self, rows, labels):
        """
        This method checks if the rows passed have are all of the same class.
        :param rows: rows
        :param labels: lables corresponding to each row
        :return: True if it is a leaf, False otherwise
        """
        n = len(labels)
        if n < self.min_for_pruning:
            return True
        count = class_counts(rows, labels)
        if (("B" in count and count["B"] == n)
                or ("M" in count and count["M"] == n)):
            return True
        return False
