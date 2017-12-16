from collections import Counter
from learn import Learner
from math import log

def entropy(labels):
    size = float(len(labels))
    counts = Counter(labels)
    return sum(-counts[k]/size * log(counts[k]/size, 2) for k in counts)

def majorityError(labels):
    majority = float(labels.count(max(labels)))
    return 1 - majority/len(labels)

class Tree(object):
    def __init__(self, value):
        self.value = value
        self.children = {}

    def addChild(self, tree):
        self.children.append(tree)

class DecisionTree(Learner):
    def __init__(self, depth=0, error=entropy):
        self.maxAllowedDepth = depth if depth > 0 else float('inf')
        self._calculateError = error

    def _calculateInformationGain(self, currentError, featureObservations, labels):
        for choice in set(featureObservations):
            reducedLabels = [label for i,label in enumerate(labels) if featureObservations[i] == choice]
            currentError -= float(len(reducedLabels))/len(labels)*self._calculateError(reducedLabels)

        return currentError

    def _getBestFeature(self, observations, labels, featureIndices):
        # Calculate error of all labels
        currentError = self._calculateError(labels)
        bestFeature = bestIG = -1

        # Compute feature with best information gain
        for feature in featureIndices:
            featureObservations = [observation.get(feature, 0) for observation in observations]
            IG = self._calculateInformationGain(currentError, featureObservations, labels)
            if IG > bestIG:
                bestIG = IG
                bestFeature = feature

        return bestFeature
        
    def _buildTree(self, observations, labels, featureIndices, depth):
        # Update max achieved depth
        if depth > self.maxAchievedDepth:
            self.maxAchievedDepth = depth

        # Base case: one label, no more features, or max depth reached -> assign majority label
        counts = Counter(labels)
        if len(counts) == 1 or not featureIndices or depth >= self.maxAllowedDepth:
            return Tree(max(counts, key=lambda k: counts[k]))

        # Branch on best feature 
        bestFeature = self._getBestFeature(observations, labels, featureIndices)
        tree = Tree(bestFeature)
        
        # Recur
        reducedFeatureIndices = featureIndices - set([bestFeature])
        bestFeatureObservations = set([observation.get(bestFeature, 0) for observation in observations])
        for choice in bestFeatureObservations:
            reducedObservations = [{k:v for k,v in observation.iteritems() if k != bestFeature} for observation in observations if observation.get(bestFeature, 0) == choice]
            reducedLabels = [label for i,label in enumerate(labels) if observations[i].get(bestFeature,0) == choice]

            # Recursive calls
            tree.children[choice] = self._buildTree(reducedObservations, reducedLabels, reducedFeatureIndices, depth + 1)

        return tree

    def train_epochs(self, observations, labels, n):
        self.train(observations, labels)
        return []

    def train(self, observations, labels):
        self.maxAchievedDepth = 1
        featureIndices = set.union(*[{k for k in sub} for sub in observations])
        self.tree = self._buildTree(observations, labels, featureIndices, 1)
        return self.maxAchievedDepth

    def test(self, observations, labels):
        results = []
        for i,observation in enumerate(observations):
            results.append((self.predict(observation), labels[i]))

        return results

    def predict(self, observation):
        tree = self.tree
        while tree.children:
            featureVal = observation.get(tree.value, 0)
            if tree.children.has_key(featureVal):
                tree = tree.children[featureVal]
            else: # Unseen feature value - we'll just go down the first child branch
                tree = tree.children[list(tree.children)[0]]

        return tree.value

