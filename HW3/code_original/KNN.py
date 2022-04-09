from utils import *


class KNNClassifier:

    def __init__(self, k=1):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None
        self.labels = None

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        :return: self
        """

        self.x_train = x_train
        self.y_train = y_train
        self.labels = list(np.unique(y_train))
        self.n_classes = len(self.labels)
        return self

    def predict(self, x_test: np.ndarray):
        """
        Predict the most likely class for each sample in a given vector.
        :param x_test: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)
        n_test = x_test.shape[0]
        y_pred = [''] * n_test
        for i in range(n_test):
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            def _nearest_k(_dist_matrix, idx, k):
                return np.transpose(_dist_matrix[:, idx:idx + 1])[0].argsort()[:k]

            top_k_indices = _nearest_k(dist_matrix, i, self.k)
            top_k_classes = self.y_train[top_k_indices]

            u, c = np.unique(top_k_classes, return_counts=True)
            pred_class = u[c.argmax()]
            y_pred[i] = pred_class

        return np.array(y_pred)
