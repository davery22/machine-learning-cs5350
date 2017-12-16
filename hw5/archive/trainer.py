from collections import Counter
from svm import SVM
from perceptron import Perceptron
from logistic_regressor import LogisticRegressor
from decision_tree import DecisionTree
from naive_bayes import NaiveBayes
from bagger import Bagger
import sys
import random
        
def extract_data(infile):
    observations = []
    labels = []

    with open(infile, 'r') as infile:
        for line in infile.readlines():
            splits = line.strip().split(' ')
            labels.append(int(splits[0]))
            splits = splits[1:] if len(splits) > 1 else []
            features = {int(kv[0]): float(kv[1]) for kv in [val.split(':') for val in splits]}
            features[0] = 1.0 # introduce bias term
            observations.append(features)

    return [observations, labels]

def get_all_data():
    train_data = extract_data('./data/speeches.train.liblinear')
    test = extract_data('./data/speeches.test.liblinear')
    train_test = 0.8
    count = len(train_data[0])
    train = [d[:int(count*train_test)] for d in train_data]
    dev = [d[int(count*train_test):] for d in train_data]

    folds = []
    for i in range(5):
        folds.append(extract_data('./data/CVSplits/training0{0}.data'.format(i)))
    return train_data, train, test, dev, folds

def majority_baseline(test, dev):
    maj_test = Counter(test[1]).most_common(1)[0][1]
    maj_dev = Counter(dev[1]).most_common(1)[0][1]
    print 'Test Accuracy: {0}/{1} ({2:.4f})'.format(maj_test, len(test[1]), float(maj_test)/len(test[1]))
    print 'Dev Accuracy: {0}/{1} ({2:.4f})'.format(maj_dev, len(dev[1]), float(maj_dev)/len(dev[1]))

def parameter_combinations(param_choices):
    if not param_choices:
        yield {}

    key = next(iter(param_choices))
    for val in param_choices[key]:
        dic = {key: val}
        remaining = dict(param_choices)
        remaining.pop(key)
        for d in parameter_combinations(remaining):
            dic.update(d)
            yield dic

def cross_validate(classtype, folds, params):
    average_acc = 0

    for k in range(len(folds)):
        sys.stdout.write('\rFold: {}/5'.format(k+1))
        sys.stdout.flush()

        test = folds[k]
        train = [[],[]]

        for j,fold in enumerate(folds):
            if j != k:
                train[0] += fold[0]
                train[1] += fold[1]

        learner = classtype(**params)
        list(learner.train_epochs(train[0], train[1], 10)) # Force the generator to finish all epochs
        results = learner.test(test[0], test[1])

        accuracy = sum(r[0] == r[1] for r in results) / float(len(results))
        average_acc += (accuracy - average_acc) / (k+1)

    print
    return average_acc

def select_hyperparams(classtype, folds, param_choices):
    best = [0, 0]

    for params in parameter_combinations(param_choices):
        average_acc = cross_validate(classtype, folds, params)
        if average_acc > best[0]:
            best[0] = average_acc
            best[1] = params
                    
    return best

def experiment1(classtype, train, test, dev, folds, param_choices={}):
    # Cross-validate to find best hyperparameters
    print 'Cross-validating to find best hyperparameters in', param_choices, '...'
    acc, params = select_hyperparams(classtype, folds, param_choices)
    print '* Best Params:', params
    print '* Best Params Average Cross-val Accuracy: {0:.4f}'.format(acc)
    
    # Train 20 epochs; pick the best weights to use later
    print 'Training on \'train\', testing against \'dev\' - for 20 epochs ...'
    best = [0, 0, 0]
    learner = classtype(**params)
    for i,weights in enumerate(learner.train_epochs(train[0], train[1], 20)):
        results = learner.test(dev[0], dev[1])
        accuracy = sum(r[0] == r[1] for r in results) / float(len(results))
        print '* Epoch {0: >2} Dev Accuracy: {1:.4f}'.format(i+1, accuracy)
        if accuracy > best[0]:
            best[0] = accuracy
            best[1] = weights
            best[2] = i+1

    print '* Train Updates:', learner.update_count

    # Use best weights from epochs to test
    learner = classtype(initial_weights=best[1], **params)

    print 'Testing against \'train\', using best epoch weights (epoch = {0})'.format(best[2])
    results = learner.test(train[0], train[1])
    accuracy = sum(r[0] == r[1] for r in results)
    print '* Train Accuracy: {0}/{1} ({2:.4f})'.format(accuracy, len(results), float(accuracy)/len(results))

    print 'Testing against \'test\', using best epoch weights (epoch = {0})'.format(best[2])
    results = learner.test(test[0], test[1])
    accuracy = sum(r[0] == r[1] for r in results)
    print '* Test Accuracy: {0}/{1} ({2:.4f})'.format(accuracy, len(results), float(accuracy)/len(results))

def experiment2(train, test, folds, param_choices={}):
    # Cross-validate to find best hyperparameters
    print 'Cross-validating to find best hyperparameters in', param_choices, '...'
    acc, params = select_hyperparams(NaiveBayes, folds, param_choices)
    print '* Best Params:', params
    print '* Best Params Average Cross-val Accuracy: {0:.4f}'.format(acc)
    
    # Use best params to test
    learner = NaiveBayes(**params)
    learner.train(train[0], train[1])

    print 'Testing against \'train\''
    results = learner.test(train[0], train[1])
    accuracy = sum(r[0] == r[1] for r in results)
    print '* Train Accuracy: {0}/{1} ({2:.4f})'.format(accuracy, len(results), float(accuracy)/len(results))

    print 'Testing against \'test\''
    results = learner.test(test[0], test[1])
    accuracy = sum(r[0] == r[1] for r in results)
    print '* Test Accuracy: {0}/{1} ({2:.4f})'.format(accuracy, len(results), float(accuracy)/len(results))

def get_trained_trees(train, treecount, samplesize):
    # Train `treecount` trees
    spots = range(len(train[1]))
    trees = []
    for i in range(treecount):
        sys.stdout.write('\r{}/{}'.format(i+1, treecount))
        sys.stdout.flush()

        # Sample `samplesize` examples with replacement
        choices = [random.choice(spots) for _ in range(samplesize)]
        example_subset = [[d[i] for i in choices] for d in train]

        # Train your tree, push it on
        tree = DecisionTree(depth=3)
        tree.train(*example_subset)
        trees.append(tree)
    
    print
    return trees

def experiment3(trees, train, test):
    # Bag'em
    learner = Bagger()
    learner.add_classifiers(trees)

    # Test
    print 'Testing ensemble against \'train\''
    results = learner.test(train[0], train[1])
    accuracy = sum(r[0] == r[1] for r in results)
    print '* Train Accuracy: {0}/{1} ({2:.4f})'.format(accuracy, len(results), float(accuracy)/len(results))

    print 'Testing ensemble against \'test\''
    results = learner.test(test[0], test[1])
    accuracy = sum(r[0] == r[1] for r in results)
    print '* Test Accuracy: {0}/{1} ({2:.4f})'.format(accuracy, len(results), float(accuracy)/len(results))

def tree_transform(trees, data):
    data[0] = [{i:tree.predict(obsv) for i,tree in enumerate(trees, start=1)} for obsv in data[0]]
    for obsv in data[0]:
        obsv[0] = 1.0 # bias term

def tree_transform_all(tree, train, test, dev, folds):
    tree_transform(trees, train)
    tree_transform(trees, test)
    tree_transform(trees, dev)
    for f in folds:
        tree_transform(trees, f)

if __name__ == '__main__':
    # Make things consistent
    random.seed(7)

    # Get all the data
    print 'Extracting data from files...'
    full_train, train, test, dev, folds = get_all_data()

    # Run experiments
    
    print '--- Majority Baseline ---\n'
    majority_baseline(test, dev)
    """
    print '\n--- 1. SVM Experiment ---\n'
    experiment1(SVM, train, test, dev, folds, param_choices={'learning_rate': [10, 1, 0.1, 0.01, 0.001, 0.0001], 'tradeoff': [10, 1, 0.1, 0.01, 0.001, 0.0001]})

    print '\n--- 2. Logistic Regression Experiment ---\n'
    experiment1(LogisticRegressor, train, test, dev, folds, param_choices={'learning_rate': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001], 'sigma_squared': [0.1, 1, 10, 100, 1000, 10000]})
    """
    print '\n--- 3. Naive Bayes Experiment ---\n'
    experiment2(full_train, test, folds, param_choices={'smoothing': [2.0, 1.5, 1.0, 0.5]})

    print '\n/\/\ ...Growing a forest for latter experiments... /\/\ \n'
    trees = get_trained_trees(full_train, 1000, 100)

    print '\n--- 4. Bagged Forest Experiment ---\n'
    experiment3(trees, full_train, test)

    tree_transform_all(trees, train, test, dev, folds)

    print '\n--- 5. SVM Over Trees Experiment ---\n'
    experiment1(SVM, train, test, dev, folds, param_choices={'learning_rate': [10, 1, 0.1, 0.01, 0.001, 0.0001], 'tradeoff': [10, 1, 0.1, 0.01, 0.001, 0.0001]})

    print '\n--- 6. Logistic Regression Over Trees Experiment ---\n'
    experiment1(LogisticRegressor, train, test, dev, folds, param_choices={'learning_rate': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001], 'sigma_squared': [0.1, 1, 10, 100, 1000, 10000]})
