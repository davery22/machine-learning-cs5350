from math import log, sqrt
from collections import Counter

class Learner(object):
    pass

class Tree(object):
    def __init__(self, value):
        self.value = value
        self.children = {}

    def addChild(self, tree):
        self.children.append(tree)

def entropy(labels):
    size = float(len(labels))
    counts = Counter(labels)
    return sum(-counts[k]/size*log(counts[k]/size, 2) for k in counts)

def majorityError(labels):
    majority = float(labels.count(max(labels)))
    return 1 - majority/len(labels)

class DecisionTree(Learner):
    def __init__(self):
        pass

    def _calculateInformationGain(self, currentError, featureObservations, labels):
        for choice in set(featureObservations):
            reducedLabels = [label for i,label in enumerate(labels) if featureObservations[i] == choice]
            currentError -= float(len(reducedLabels))/len(labels)*self._calculateError(reducedLabels)

        return currentError

    def _getBestFeature(self, observations, labels, featureIndices):
        currentError = self._calculateError(labels)
        bestFeatureGlobal = bestFeatureLocal = bestIG = -1
        for feature in range(len(featureIndices)):
            featureObservations = [observation[feature] for observation in observations]
            IG = self._calculateInformationGain(currentError, featureObservations, labels)
            if IG > bestIG:
                bestIG = IG
                bestFeatureLocal = feature
                bestFeatureGlobal = featureIndices[feature]

        return bestFeatureGlobal, bestFeatureLocal
        
    def _buildTree(self, observations, labels, featureIndices, depth):
        if depth > self.maxAchievedDepth:
            self.maxAchievedDepth = depth

        # Base case: one label, no more features, or max depth reached -> assign majority label
        counts = Counter(labels)
        if len(counts) == 1 or not observations[0] or depth >= self.maxDepth:
            return Tree(max(counts, key=lambda k: counts[k]))

        bfg, bfl = self._getBestFeature(observations, labels, featureIndices)
        tree = Tree(bfg)
        reducedFeatureIndices = [el for el in featureIndices if el != bfg]
        
        for choice in set([observation[bfl] for observation in observations]):
            reducedObservations = [[el for i,el in enumerate(observation) if i != bfl] for observation in observations if observation[bfl] == choice]
            reducedLabels = [label for i,label in enumerate(labels) if observations[i][bfl] == choice]
            tree.children[choice] = self._buildTree(reducedObservations, reducedLabels, reducedFeatureIndices, depth + 1)

        return tree

    def train(self, observations, labels, maxDepth=-1, error=entropy):
        self.maxDepth = maxDepth if maxDepth > 0 else float('inf')
        self.maxAchievedDepth = 1
        self._calculateError = error
        self.tree = self._buildTree(observations, labels, range(len(observations[0])), 1)
        return self.maxAchievedDepth

    def test(self, observations, labels):
        results = []
        for i,observation in enumerate(observations):
            results.append((self.predict(observation), labels[i]))

        return results

    def predict(self, observation):
        tree = self.tree
        while tree.children:
            feature = observation[tree.value]
            if tree.children.has_key(feature):
                tree = tree.children[feature]
            else: # Unseen feature value - we'll just go down the first child branch
                tree = tree.children[list(tree.children)[0]]

        return tree.value

'''
END OF DECISION TREE IMPLEMENTATION
START OF EXPERIMENTS
'''

def featurize(rawObservation):
    splitname = rawObservation.split(' ')
    vowelCount = sum([len([el for el in split if el in 'aeiouAEIOU']) for split in splitname])
    consonantCount = len(''.join(splitname)) - vowelCount

    # Suggested features
    firstLongerLast = len(splitname[0]) > len(splitname[-1])
    haveMiddle = len(splitname) > 2
    firstStartEndSame = splitname[0].lower()[0] == splitname[0].lower()[-1]
    firstBeforeLast = splitname[0] < splitname[-1]
    secondOfFirstVowel = splitname[0][1] in 'aeiouAEIOU' if len(splitname[0]) > 1 else False
    lastEven = len(splitname[-1]) % 2 == 0

    # Made-up features
    longestPartIsLong = len(max(splitname, key=len)) > 6
    shortestPartIsShort = len(min(splitname, key=len)) < 3
    evenVowelCount = vowelCount % 2 == 0
    over4Vowels = vowelCount > 4
    shortName = len(''.join(splitname)) < 8
    longName = len(''.join(splitname)) > 13
    abbreviated = '.' in ''.join(splitname)
    earlyLastName = splitname[-1] < 'M'
    earlyFirstName = splitname[0] < 'M'
    firstLastStartSame = splitname[0][0] == splitname[-1][0]
    firstLastEndSame = splitname[0][-1] == splitname[-1][-1]
    evenConsonantCount = consonantCount % 2 == 0
    over6Consonants = consonantCount > 6
    sameConsonantsVowels = consonantCount == vowelCount

    return [
        firstLongerLast, haveMiddle, firstStartEndSame, firstBeforeLast, secondOfFirstVowel, lastEven,
        longestPartIsLong, shortestPartIsShort, evenVowelCount, over4Vowels, shortName, longName, abbreviated, earlyLastName, earlyFirstName, firstLastStartSame,
        firstLastEndSame, evenConsonantCount, over6Consonants, sameConsonantsVowels
    ]

def getDataFromFile(dataFile):
    with open(dataFile, 'r') as dataFile:
        lines = [line.strip() for line in dataFile.readlines()]
    
    labels = [line[0] for line in lines]
    observations = [featurize(line[2:]) for line in lines]

    return observations, labels

def basicRun(maxDepth=-1):
    trainFile = './Updated_Dataset/updated_train.txt'
    observations, labels = getDataFromFile(trainFile)

    classifier = DecisionTree()
    depth = classifier.train(observations, labels, maxDepth=maxDepth)

    print 'Achieved Depth: {0}'.format(depth)
    results = classifier.test(observations, labels)
    #print 'Train results (predicted, actual)'
    #print results
    countCorrect = len([i for i in results if i[0] == i[1]])
    print 'Train Accuracy: {0} / {1} ({2:.4f})'.format(countCorrect, len(results), float(countCorrect)/len(results))

    testFile = './Updated_Dataset/updated_test.txt'
    observations, labels = getDataFromFile(testFile)

    results = classifier.test(observations, labels)
    #print 'Test Results (predicted, actual)'
    #print results
    countCorrect = len([i for i in results if i[0] == i[1]])
    print 'Test Accuracy: {0} / {1} ({2:.4f})'.format(countCorrect, len(results), float(countCorrect)/len(results))
    print

def crossValRun():
    folds = []
    for i in range(4):
        folds.append(getDataFromFile('./Updated_Dataset/Updated_CVSplits/updated_training0{0}.txt'.format(i)))

    print '{0: ^18} {1: ^18} {2: ^18}'.format('max_depth', 'avg_accuracy', 'std_dev_accuracy')
    stats = []
    classifier = DecisionTree()
    for i in [1,2,3,4,5,10,15,20]:
        accuracies = []
        for k in range(4):
            test = folds[k]
            train = [[],[]]
            for j,fold in enumerate(folds):
                if j != k:
                    train[0] += fold[0]
                    train[1] += fold[1]

            classifier.train(train[0], train[1], maxDepth=i)
            results = classifier.test(test[0], test[1])
            accuracy = len([r for r in results if r[0] == r[1]]) / float(len(results))
            accuracies.append(accuracy)

        avg = sum(accuracies)/float(len(accuracies))
        stdDev = sqrt(sum(map(lambda x: (x - avg)**2, accuracies)) / float(len(accuracies)))
        stats.append((i, avg, stdDev))
        print '{0: >18} {1: >18.4f} {2: >18.4f}'.format(i, avg, stdDev)

    print
    bestDepth = max(stats, key=lambda stat: stat[1])[0]
    return bestDepth

if __name__ == '__main__':
    print '--- Running with unconstrained depth ---\n'
    basicRun()
    print '--- Running cross-validation tests ---\n'
    bestDepth = crossValRun()
    print '--- Running with best empirical depth limit (depth = {0}) ---\n'.format(bestDepth)
    basicRun(bestDepth)

