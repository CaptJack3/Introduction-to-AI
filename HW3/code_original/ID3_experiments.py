from ID3 import ID3
from utils import *

"""
Make the imports of python packages needed
"""

"""
========================================================================
========================================================================
                              Experiments 
========================================================================
========================================================================
"""
target_attribute = 'diagnosis'


def find_best_pruning_m(train_dataset: np.array, m_choices, num_folds=5):
    """
    Use cross validation to find the best M for the id3 model.

    :param train_dataset: Training dataset.
    :param m_choices: A sequence of possible value of M for the ID3 model min_for_pruning attribute.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_M, accuracies) where:
        best_M: the value of M with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each M (list of lists).
    """

    accuracies = []
    for i, m in enumerate(m_choices):
        model = ID3(label_names=attributes_names, min_for_pruning=m)
        # TODO:
        #  - Add a KFold instance of sklearn.model_selection, pass <ID> as random_state argument.
        #  - Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use create_train_validation_split train/validation splitter from utils.py
        #  (note that then it won't be exactly k-fold CV since it will be a random split each iteration),
        #  or implement something else.

        # ====== YOUR CODE: ======
        raise NotImplementedError
        # ========================

    best_m_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_m = m_choices[best_m_idx]

    return best_m, accuracies


# ========================================================================
def basic_experiment(x_train, y_train, x_test, y_test, formatted_print=False):
    """
    Use ID3 model, to train on the training dataset and evaluating the accuracy in the test set.
    """

    # TODO:
    #  - Instate ID3 decision tree instance.
    #  - Fit the tree on the training data set.
    #  - Test the model on the test set (evaluate the accuracy) and print the result.

    acc = None

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================

    assert acc > 0.9, 'you should get an accuracy of at least 90% for the full ID3 decision tree'
    print(f'Test Accuracy: {acc * 100:.2f}%' if formatted_print else acc)


# ========================================================================
def cross_validation_experiment(plot_graph=True):
    """
    Use cross validation to find the best M for the ID3 model, used as pruning parameter.

    :param plot_graph: either to plot or not the experiment result, default is True
    :return: best_m: the value of M with the highest mean accuracy across folds
    """
    # TODO:
    #  - fill the m_choices list with  at least 5 different values for M.
    #  - Instate ID3 decision tree instance.
    #  - Fit the tree on the training data set.
    #  - Test the model on the test set (evaluate the accuracy) and print the result.
    best_m = None
    accuracies = []
    m_choices = []
    num_folds = 5
    if len(m_choices) < 5:
        print('fill the m_choices list with  at least 5 different values for M.')
        return None

    # ====== YOUR CODE: ======

    # ========================
    accuracies_mean = np.array([np.mean(acc) * 100 for acc in accuracies])
    if best_m is not None and plot_graph:
        util_plot_graph(x=m_choices, y=accuracies_mean, x_label='M', y_label='Validation Accuracy %')
        print('{:^10s} | {:^10s}'.format('M value', 'Validation Accuracy'))
        for i, m in enumerate(m_choices):
            print('{:^10d} | {:.2f}%'.format(m, accuracies_mean[i]))
        print(f'===========================')
        # Calculate accuracy
        accuracy_best_m = accuracies_mean[m_choices.index(best_m)]
        print('{:^10s} | {:^10s}'.format('Best M', 'Validation Accuracy'))
        print('{:^10d} | {:.2f}%'.format(best_m, accuracy_best_m))

    # ========================
    return best_m

    """
    Use cross validation to find the best M for the ID3 model, used as pruning parameter.

    :param plot_graph: either to plot or not the experiment result, default is True
    :return: best_m: the value of M with the highest mean accuracy across folds
    """
    # TODO:
    #  - fill the m_choices list with  at least 5 different values for M.
    #  - Instate ID3 decision tree instance.
    #  - Fit the tree on the training data set.
    #  - Test the model on the test set (evaluate the accuracy) and print the result.

    best_m = None
    accuracies = []
    m_choices = []
    num_folds = 5

    # ====== YOUR CODE: ======
    assert len(m_choices) >= 5, 'fill the m_choices list with  at least 5 different values for M.'
    

    # ========================
    accuracies_mean = np.array([np.mean(acc) * 100 for acc in accuracies])
    if len(m_choices) >= 5 and plot_graph:
        util_plot_graph(x=m_choices, y=accuracies_mean, x_label='M', y_label='Validation Accuracy %')
        print('{:^10s} | {:^10s}'.format('M value', 'Validation Accuracy'))
        for i, m in enumerate(m_choices):
            print('{:^10d} | {:.2f}%'.format(m, accuracies_mean[i]))
        print(f'===========================')
        # Calculate accuracy
        accuracy_best_m = accuracies_mean[m_choices.index(best_m)]
        print('{:^10s} | {:^10s}'.format('Best M', 'Validation Accuracy'))
        print('{:^10d} | {:.2f}%'.format(best_m, accuracy_best_m))

    return best_m


# ========================================================================
def best_m_test(x_train, y_train, x_test, y_test, min_for_pruning):
    """
        Test the pruning for the best M value we have got from the cross validation experiment.
        :param: best_m: the value of M with the highest mean accuracy across folds
        :return: acc: the accuracy value of ID3 decision tree instance that using the best_m as the pruning parameter.
    """

    # TODO:
    #  - Instate ID3 decision tree instance (using pre-training pruning condition).
    #  - Fit the tree on the training data set.
    #  - Test the model on the test set (evaluate the accuracy) and return the result.

    acc = None

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================

    return acc


# ========================================================================
if __name__ == '__main__':
    attributes_names, train_dataset, test_dataset = load_data_set('ID3')
    data_split = get_dataset_split(train_dataset, test_dataset, target_attribute)

    """
    Usages helper:
    (*) To get the results in “informal” or nicely printable string representation of an object
        modify the call "utils.set_formatted_values(value=False)" from False to True and run it
    """
    formatted_print = True
    basic_experiment(*data_split, formatted_print)

    """
       cross validation experiment
       (*) To run the cross validation experiment over the  M pruning hyper-parameter 
           uncomment below code and run it
           modify the value from False to True to plot the experiment result
    """
    plot_graphs = True
    best_m = cross_validation_experiment(plot_graph=plot_graphs) or 50
    print(f'best_m = {best_m}')

    """
        pruning experiment, run with the best parameter
        (*) To run the experiment uncomment below code and run it
    """
    acc = best_m_test(*data_split, min_for_pruning=best_m)
    assert acc > 0.95, 'you should get an accuracy of at least 95% for the pruned ID3 decision tree'
    print(f'Test Accuracy: {acc * 100:.2f}%' if formatted_print else acc)
