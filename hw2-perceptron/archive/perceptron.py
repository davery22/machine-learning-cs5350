import random
from collections import Counter

def new_weight():
    return random.uniform(-0.01, 0.01)

def dense_sparse_dot_product(arr, dic):
    total = 0
    for i,el in enumerate(arr):
        total += el * (dic[i] if i in dic else 0)
    return total

def sparse_sparse_dot_product(dic1, dic2):
    total = 0
    for i in dic1:
        if i in dic2:
            total += dic1[i] * dic2[i]
    return total

class Perceptron(object):
    '''
    The Perceptron class encapsulates all the variants of perceptron, which are configured via keyword arguments to the constructor.
    '''

    def __init__(self, learning_rate=1, dynamic_learning=False, margin=0, aggressive=False, average=False, initial_weights=[]):
        self.error_count = 0
        self.learning_rate = float(learning_rate)
        self.dynamic_learning = dynamic_learning
        self.margin=margin
        self.aggressive=aggressive
        self.average=average
        self.seen = 0
        self.weights = initial_weights[:]
        if self.average:
            self.average_weights = initial_weights[:]

    def train(self, observations, labels):
        for i,_ in enumerate(observations):
            self.train_online(observations[i], labels[i], ret=False)
        return self.average_weights[:] if self.average else self.weights[:]

    def train_epochs(self, observations, labels, epochs):
        zipped = zip(observations, labels)
        for _ in range(epochs):
            random.shuffle(zipped)
            yield self.train(*zip(*zipped))

    def train_online(self, observation, label, ret=True):
        self.seen += 1
        self._ensure_weights(observation)

        # Check the update condition.
        if label * dense_sparse_dot_product(self.weights, observation) < self.margin:
            self._update_weights(observation, label)

        # Incrementally update the average weight vector. Expensive, but necessary to preserve online compatibility.
        if self.average:
            for i,_ in enumerate(self.weights):
                self.average_weights[i] += (self.weights[i] - self.average_weights[i]) / (self.seen)
            
        if ret:
            return self.average_weights[:] if self.average else self.weights[:]

    def _update_weights(self, observation, label):
        # Learning rate priority: aggressive > dynamic_learning > default
        if self.aggressive:
            learning_rate = (self.margin - label * dense_sparse_dot_product(self.weights, observation)) / (sparse_sparse_dot_product(observation, observation) + 1)
        elif self.dynamic_learning:
            learning_rate = self.learning_rate / float(1 + self.error_count)
        else:
            learning_rate = self.learning_rate

        self.error_count += 1
        for i in observation:
            self.weights[i] += learning_rate * label * observation[i]

    def _ensure_weights(self, observation):
        # The price we pay for using sparse observations - less memory and more adaptable, but have to make sure the weight vector is big enough every time.
        max_index = max(observation)
        while len(self.weights) < max_index+1:
            self.weights.append(new_weight())
            if self.average:
                self.average_weights.append(0.0)

    def test(self, observations, labels):
        results = []
        for i,_ in enumerate(observations):
            results.append((self.predict(observations[i]), labels[i]))
        return results

    def predict(self, observation):
        self._ensure_weights(observation)
        weights = self.average_weights if self.average else self.weights
        if dense_sparse_dot_product(weights, observation) >= 0:
            return 1
        else:
            return -1
        
def extract_data(infile):
    observations = []
    labels = []

    with open(infile, 'r') as infile:
        for line in infile.readlines():
            splits = line.split(' ')
            labels.append(int(splits[0]))
            splits = splits[1:] if len(splits) > 1 else []
            features = {int(kv[0]): float(kv[1]) for kv in [val.split(':') for val in splits]}
            features[0] = 1.0 # introduce bias term
            observations.append(features)

    return observations, labels

def get_all_data():
    train = extract_data('./Dataset/phishing.train')
    test = extract_data('./Dataset/phishing.test')
    dev = extract_data('./Dataset/phishing.dev')
    folds = []
    for i in range(5):
        folds.append(extract_data('./Dataset/CVSplits/training0{0}.data'.format(i)))
    return train, test, dev, folds

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

def cross_validate(folds, params):
    average_acc = 0

    for k in range(len(folds)):
        test = folds[k]
        train = [[],[]]

        for j,fold in enumerate(folds):
            if j != k:
                train[0] += fold[0]
                train[1] += fold[1]

        learner = Perceptron(**params)
        list(learner.train_epochs(train[0], train[1], 10)) # Force the generator to finish all epochs
        results = learner.test(test[0], test[1])

        accuracy = sum(r[0] == r[1] for r in results) / float(len(results))
        average_acc += (accuracy - average_acc) / (k+1)

    return average_acc

def select_hyperparams(folds, param_choices):
    best = [0, 0]

    for params in parameter_combinations(param_choices):
        average_acc = cross_validate(folds, params)
        if average_acc > best[0]:
            best[0] = average_acc
            best[1] = params
                    
    return best

def experiment(train, test, dev, folds, param_choices={}):
    print 'Cross-validating to find best hyperparameters in', param_choices, '...'
    acc, params = select_hyperparams(folds, param_choices)
    print '* Best Cross-val Accuracy: {0:.4f}'.format(acc)
    print '* Best Params:', params
    
    print 'Training on \'train\', testing against \'dev\' - for 20 epochs ...'
    best = [0, 0, 0]
    learner = Perceptron(**params)
    for i,weights in enumerate(learner.train_epochs(train[0], train[1], 20)):
        results = learner.test(dev[0], dev[1])
        accuracy = sum(r[0] == r[1] for r in results) / float(len(results))
        print '* Epoch {0: >2} Dev Accuracy: {1:.4f}'.format(i+1, accuracy)
        if accuracy > best[0]:
            best[0] = accuracy
            best[1] = weights
            best[2] = i+1

    print '* Train Updates:', learner.error_count

    print 'Testing against \'test\', using best epoch weights (epoch = {0})'.format(best[2])
    learner = Perceptron(initial_weights=best[1], **params)
    results = learner.test(test[0], test[1])
    accuracy = sum(r[0] == r[1] for r in results)

    print '* Test Accuracy: {0}/{1} ({2:.4f})'.format(accuracy, len(results), float(accuracy)/len(results))

if __name__ == '__main__':
    # Make things consistent
    random.seed(7)

    # Get all the data
    train, test, dev, folds = get_all_data()

    # Run experiments
    print '--- Majority Baseline ---\n'
    majority_baseline(test, dev)
    print '\n--- 1. Simple Perceptron Experiment ---\n'
    experiment(train, test, dev, folds, param_choices={'learning_rate': [1, 0.1, 0.01]})
    print '\n--- 2. Dynamic Learning Rate Perceptron Experiment ---\n'
    experiment(train, test, dev, folds, param_choices={'learning_rate': [1, 0.1, 0.01], 'dynamic_learning': [True]})
    print '\n--- 3. Margin Perceptron Experiment ---\n'
    experiment(train, test, dev, folds, param_choices={'learning_rate': [1, 0.1, 0.01], 'dynamic_learning': [True], 'margin': [1, 0.1, 0.01]})
    print '\n--- 4. Averaged Perceptron Experiment ---\n'
    experiment(train, test, dev, folds, param_choices={'learning_rate': [1, 0.1, 0.01], 'average': [True]})
    print '\n--- 5. Aggressive Perceptron with Margin Experiment ---\n'
    experiment(train, test, dev, folds, param_choices={'margin': [1, 0.1, 0.01], 'aggressive': [True]})

