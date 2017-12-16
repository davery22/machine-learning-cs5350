class Bagger(object):
    def __init__(self):
        self.classifiers = []

    def add_classifiers(self, classifiers):
        self.classifiers += classifiers
        
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
