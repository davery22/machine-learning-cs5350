from collections import Counter
from learn import Learner
import random

def new_weight():
    return random.uniform(-0.01, 0.01)

def dense_sparse_dot_product(arr, dic):
    total = 0
    for i in dic:
        total += arr[i] * dic[i]
    return total

def sparse_sparse_dot_product(dic1, dic2):
    total = 0
    for i in dic1:
        if i in dic2:
            total += dic1[i] * dic2[i]
    return total

class Perceptron(Learner):
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

