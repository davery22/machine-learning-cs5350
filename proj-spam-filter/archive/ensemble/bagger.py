class Bagger(object):
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def train(self, observations, labels):
        pass

    def test(self, observations, labels):
        results = []
        for i,o in enumerate(observations):
            results.append((self.predict(o), labels[i]))

        return results

    def predict(self, observation):
        votes = []
        for c in self.classifiers:
            votes.append(c.predict(observation))

        return max(set(votes), key=votes.count)
