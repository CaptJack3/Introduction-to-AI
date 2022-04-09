from pickletools import UP_TO_NEWLINE
from turtle import down
import unittest
import math
import numpy as np
from DecisonTree import Question, class_counts
from ID3 import ID3
from KNN import KNNClassifier
from utils import accuracy, l2_dist, load_data_set, get_dataset_split
from ID3_experiments import basic_experiment


def load_i3d_data():
    attributes_names, train_dataset, test_dataset = load_data_set("ID3")
    data_split = get_dataset_split(train_dataset, test_dataset, "diagnosis")
    (x_train, y_train, x_test, y_test) = data_split
    return x_train, y_train, x_test, y_test, attributes_names


def load_small_train_set(n_rows, n_features):
    x_train, y_train, _, _, _ = load_i3d_data()
    small_x_train = x_train[:n_rows, :n_features]
    small_y_train = y_train[:n_rows]
    return small_x_train, small_y_train


class Test(unittest.TestCase):

    def setUp(self):
        method = self._testMethodName
        print(method + f":{' ' * (25 - len(method))}", end='')

    def test_l2_dist_custom(self):
        a = np.array([[1, 1], [2, 2]])
        b = np.array([[2, 0], [2, 2]])
        matrix = np.array([a, b])
        self.assertTrue(
            np.array_equal(l2_dist(a, b), [[np.sqrt(2), np.sqrt(2)], [2, 0]]))
        print("Success")

    def test_accuracy_custom(self):
        c = np.array([1, 2, 3, 4])
        d = np.array([1, 2, 2, 4])
        self.assertEqual(accuracy(c, d), 0.75)
        print("Success")

    def test_class_counts_id3(self):
        small_x_train, small_y_train = load_small_train_set(5, 3)
        train_counts = class_counts(small_x_train,
                                    small_y_train)  # returns a dictionary
        B_counts = train_counts["B"]
        M_counts = train_counts["M"]
        self.assertEqual((B_counts, M_counts), (2, 3))
        print("Success")

    def test_entropy(self):
        n_rows = 15
        # set manually the numbers of B and of M in the first "n_rows" of the
        # train set
        n_B = 9
        n_M = 6
        prob_B = n_B / n_rows
        prob_M = n_M / n_rows
        small_x_train, small_y_train = load_small_train_set(n_rows, 3)
        impurity = ID3.entropy(small_x_train, small_y_train)
        expected_imurity = \
            - prob_B * math.log(prob_B, 2) \
            - prob_M * math.log(prob_M, 2)
        self.assertAlmostEqual(impurity, expected_imurity, places=1)
        print("Success")

    def test_entropy_leaf_case(self):
        n_rows = 2
        small_x_train, small_y_train = load_small_train_set(n_rows, 3)
        impurity = ID3.entropy(small_x_train, small_y_train)
        expected_impurity = 0
        self.assertEqual(impurity, expected_impurity)
        print("Success")

    def test_info_gain(self):
        search_tree = ID3(["B", "M"])
        small_x_train, small_y_train = load_small_train_set(5, 2)
        current_uncertainty = search_tree.entropy(small_x_train,
                                                  small_y_train)
        # separation
        left_rows = small_x_train[:3]
        left_labels = small_y_train[:3]
        right_rows = small_x_train[3:]
        right_labels = small_y_train[3:]
        i_gain = search_tree.info_gain(left_rows, left_labels,
                                       right_rows, right_labels,
                                       current_uncertainty)

        # computed manually
        expected_gain = 0.970950594 - 0.5509705
        self.assertAlmostEqual(i_gain, expected_gain, places=3)
        print("Success")

    def test_partition(self):
        small_x_train, small_y_train = load_small_train_set(5, 2)
        search_tree = ID3(["B", "M"])
        question = Question("radius mean", 0, 13.0)
        gain, true_rows, true_labels, false_rows, false_labels = \
            search_tree.partition(small_x_train, small_y_train, question,
                                  ID3.entropy(small_x_train, small_y_train))
        condition_on_rows_len = (len(true_rows) == len(true_labels) == 2) \
            and (len(false_rows) == len(false_labels) == 3)

        self.assertTrue(condition_on_rows_len)
        print("Success")

    def test_find_best_split(self):
        small_x_train, small_y_train = load_small_train_set(5, 1)
        search_tree = ID3(["B", "M"])
        _, question, best_true_rows, best_true_labels, _, _ = \
            search_tree.find_best_split(small_x_train, small_y_train)

        self.assertEqual(
            3, np.count_nonzero(np.equal(best_true_rows, small_x_train[2:5])))
        self.assertEqual(
            3, np.count_nonzero(np.equal(best_true_labels,
                                         small_y_train[2:5])))
        print("Success")

    def test_partition_equal_best_split(self):
        # results using find_best_split
        small_x_train, small_y_train = load_small_train_set(20, 20)
        search_tree = ID3(["B", "M"])
        _, question, best_true_rows, best_true_labels, _, _ = \
            search_tree.find_best_split(small_x_train, small_y_train)

        # results of split from partition
        _, true_rows, true_labels, _, _ = \
            search_tree.partition(small_x_train, small_y_train, question,
                                  ID3.entropy(small_x_train, small_y_train))

        n_rows_elements = true_rows.size
        n_label_elements = true_labels.size
        self.assertEqual(n_rows_elements, \
            np.count_nonzero(np.equal(best_true_rows, true_rows)))
        self.assertTrue(n_label_elements, \
            np.count_nonzero(np.compare_chararrays(
                best_true_labels, true_labels, "==", True)))
        print("Success")

    def test_is_leaf(self):
        small_x_train, small_y_train = load_small_train_set(5, 1)
        search_tree = ID3(["B", "M"])
        _, _, true_rows, true_labels, false_rows, false_labels = \
            search_tree.find_best_split(small_x_train, small_y_train)
        # with these paramenters, true_labels is a numpy.array of length 3
        # and with three 'M' as elements
        condition = (search_tree.is_leaf(true_rows, true_labels)
                     and search_tree.is_leaf(false_rows, false_labels))
        self.assertTrue(condition)

    def test_basic_experiment_without_early_pruning(self):
        attributes_names, train_dataset, test_dataset = load_data_set('ID3')
        data_split = get_dataset_split(train_dataset, test_dataset,
                                       "diagnosis")
        formatted_print = True
        print(f"Basic Experiment without early pruning:")
        acc = basic_experiment(*data_split,
                               formatted_print,
                               attributes_names,
                               test_instance=True)
        expected_result_rounded = 0.946902
        # expected_result_rounded = 0.973451
        self.assertAlmostEqual(acc, expected_result_rounded, places=5)
        print("Success")

    # def test_basic_experiment_with_early_pruning(self):
    #     attributes_names, train_dataset, test_dataset = load_data_set('ID3')
    #     data_split = get_dataset_split(train_dataset, test_dataset,
    #                                    "diagnosis")
    #     formatted_print = True
    #     print(f"Basic Experiment without early pruning:")
    #     acc = basic_experiment(*data_split,
    #                            formatted_print,
    #                            attributes_names,
    #                            test_instance=True)
    #     # expected_result_rounded = 0.946902
    #     expected_result_rounded = 0.973451
    #     self.assertAlmostEqual(acc, expected_result_rounded, places=5)
    #     print("Success")

    #####################################
    # Custom tests UP
    # Preexistent tests DOWN
    #####################################

    def test_l2_dist_shape(self):
        mat1 = np.array([[1, 1, 1], [2, 2, 2]])  # (N1, D) = (2, 3)
        mat2 = np.array([[0, 0, 0]])  # (N2, D) = (1, 3)

        l2_dist_result = l2_dist(mat1, mat2)

        self.assertEqual(l2_dist_result.shape, (2, 1))
        print('Success')

    def test_l2_dist_result(self):
        mat1 = np.array([[1, 1, 1], [2, 2, 2]])  # (N1, D) = (2, 3)
        mat2 = np.array([[0, 0, 0]])  # (N2, D) = (1, 3)

        l2_dist_result = l2_dist(mat1, mat2)
        self.assertTrue(np.array_equal(l2_dist_result, np.sqrt([[3], [12]])))
        print('Success')

    def test_accuracy(self):
        y1 = np.array([1, 1, 1])  # (N1, D) = (3,)
        y2 = np.array([1, 0, 0])  # (N2, D) = (3,)

        accuracy_val = accuracy(y1, y2)
        self.assertEqual(accuracy_val, 1 / 3)
        print('Success')

    def test_knn1(self):
        x_train = np.array([[1, 1], [2, 1]])
        y_train = np.array(['+', '-'])
        x_test = np.array([[3, 1]])
        y_test = np.array(['+'])
        classifier = KNNClassifier(k=1)
        classifier.train(x_train=x_train, y_train=y_train)
        y_pred = classifier.predict(x_test)

        accuracy_val = accuracy(y_test, y_pred)
        self.assertEqual(accuracy_val, 0)
        self.assertFalse(np.array_equal(y_pred, y_test))
        print('Success')

    def test_knn2(self):
        x_train = np.array([[1, 1], [2, 1], [0, 1]])
        y_train = np.array(['+', '-', '-'])
        x_test = np.array([[3, 1], [2, 2]])
        y_test = np.array(['-', '-'])
        classifier = KNNClassifier(k=3)
        classifier.train(x_train=x_train, y_train=y_train)
        y_pred = classifier.predict(x_test)

        accuracy_val = accuracy(y_test, y_pred)
        self.assertEqual(accuracy_val, 1)
        self.assertTrue(np.array_equal(y_pred, y_test))
        print('Success')

    def test_knn3(self):
        x_train = np.array([[1], [7], [8], [9]])
        y_train = np.array(['+', '-', '$', '$'])
        x_test = np.array([[19], [20]])
        y_test = np.array(['-', '$'])
        classifier = KNNClassifier(k=4)
        classifier.train(x_train=x_train, y_train=y_train)
        y_pred = classifier.predict(x_test)

        accuracy_val = accuracy(y_test, y_pred)
        self.assertEqual(accuracy_val, 1 / 2)
        print('Success')


if __name__ == '__main__':
    unittest.main()
