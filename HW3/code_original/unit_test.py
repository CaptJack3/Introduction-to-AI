import unittest
import numpy as np
from KNN import KNNClassifier
from utils import accuracy, l2_dist


class Test(unittest.TestCase):

    def setUp(self):
        method = self._testMethodName
        print(method + f":{' ' * (25 - len(method))}", end='')

    def test_l2_dist_shape(self):
        mat1 = np.array([[1, 1, 1],
                         [2, 2, 2]])  # (N1, D) = (2, 3)
        mat2 = np.array([[0, 0, 0]])  # (N2, D) = (1, 3)

        l2_dist_result = l2_dist(mat1, mat2)

        self.assertEqual(l2_dist_result.shape, (2, 1))
        print('Success')

    def test_l2_dist_result(self):
        mat1 = np.array([[1, 1, 1],
                         [2, 2, 2]])  # (N1, D) = (2, 3)
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
