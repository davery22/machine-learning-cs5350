from math import log, exp

class AdaBoost(object):
    def __init__(self, iters=10, epochs=10, dev=None, classtype=None, params=None):
        self.iters = iters
        self.epochs = epochs
        self.dev = dev
        self.classtype = classtype
        self.params = params
        self.classifiers = []
        self.classifier_weights = []

    def train_epochs(self, observations, labels, epochs):
        self.train(self, observations, labels)
        return []

    def train(self, observations, labels):
        # Construct initial weight distribution over examples (sums to 1)
        count = len(observations)
        dist = [1.0/count for _ in observations]

        # Train `iters` classifiers
        for _ in range(self.iters):
            # Find best params (cross-validate), find best weights (epochs)
            best = [0,0]
            classifier = self.classtype(**self.params) if self.params else self.classtype()
            for i,weights in enumerate(classifier.train_epochs_weighted(observations, labels, dist, self.epochs)):
                results = classifier.test(self.dev[0], self.dev[1])
                accuracy = sum(r[0] == r[1] for r in results) / float(len(results))
                if accuracy > best[0]:
                    best[0] = accuracy
                    best[1] = weights

            # Constuct new classifier with best weights
            print best[0]
            classifier = self.classtype(initial_weights=best[1])

            # Get alpha value
            accuracy = sum([r[0] == r[1] for r in classifier.test(observations, labels)]) / float(count)
            alpha = 0.5 * log(accuracy / (1 - accuracy))

            # Add classifier and its weight onto the ensemble
            self.classifiers.append(classifier)
            self.classifier_weights.append(alpha)

            # Construct next weight distribution
            Z = 0.0
            for i in range(count):
                dist[i] *= exp(-alpha*labels[i]*classifier.predict(observations[i]))
                Z += dist[i]
            for i in range(count):
                dist[i] /= Z

    def test(self, observations, labels):
        results = []
        for i,_ in enumerate(observations):
            results.append((self.predict(observations[i]), labels[i]))
        return results

    def predict(self, observation):
        summ = sum(map(lambda x,y: x*y, self.classifier_weights, [c.predict(observation) for c in self.classifiers]))
        return 1 if summ >= 0 else -1

