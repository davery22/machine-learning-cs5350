from learn import Learner
from collections import Counter
from math import log, exp

class NaiveBayes(Learner):
    def __init__(self, smoothing=1.0):
        self.smoothing = float(smoothing)

    def train_epochs(self, observations, labels, n):
        self.train(observations, labels)
        return []

    def train(self, observations, labels):
        # NOTE: ASSUMES FEATURES ARE BINARY (for speedup).
        labelcount = len(labels)
        counter = Counter(labels)
        self.prior = {l:log(float(c)/labelcount) for l,c in counter.iteritems()}
        self.featuresLength = max(set.union(*[set(o) for o in observations])) + 1
        self.likelihoods = {k:{i:0 for i in range(self.featuresLength)} for k in counter}

        # Count all the feature values given the label
        for label in counter:
            reducedObservations = [obsv for i,obsv in enumerate(observations) if labels[i] == label]
            for o in reducedObservations:
                for feature in o:
                    self.likelihoods[label][feature] += 1

        # Compute probability for each feature value given each label
        for label,features in self.likelihoods.iteritems():
            for feature,value in features.iteritems():
                self.likelihoods[label][feature] = log((self.likelihoods[label][feature] + self.smoothing) / (counter[label] + 2*self.smoothing))
            
    def test(self, observations, labels):
        results = []
        for i,o in enumerate(observations):
            results.append((self.predict(o), labels[i]))
        return results

    def _posterior(self, observation, label):
        prod = self.prior[label]
        for feature in range(self.featuresLength):
            pos_likelihood = self.likelihoods[label].get(feature, log(0.000001)) # Small number
            prod += pos_likelihood if observation.has_key(feature) else log(1 - exp(pos_likelihood))
        return prod

    def predict(self, observation):
        probs = {}
        for label in self.likelihoods:
            probs[label] = self._posterior(observation, label)

        return max(probs, key=probs.get)
