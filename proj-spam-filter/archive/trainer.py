from collections import Counter
from classifiers import decisiontree
import sys
import random
        
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
        sys.stdout.write('Fold: {}/{}\n'.format(k+1, len(folds)))
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
        print accuracy
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

def get_trained_models(classtype, train, dev, param_choices, modelcount, samplesize=None, featurelimit=None):
    example_spots = range(len(train[0]))
    feature_spots = range(max(map(max, train[0])))
    models = []
    param_choices = list(parameter_combinations(param_choices))
    for i in range(modelcount):
        sys.stdout.write('\r{}/{}'.format(i+1, modelcount))
        sys.stdout.flush()

        # Sample `samplesize` examples with replacement, `featurelimit` features from each example
        example_choices = [random.choice(example_spots) for _ in range(samplesize)] if samplesize else example_spots
        feature_choices = set(random.sample(feature_spots, featurelimit)) if featurelimit else feature_spots
        subset_train = [[{k:v for k,v in train[0][i].iteritems() if k in feature_choices} for i in example_choices], [train[1][i] for i in example_choices]]

        # Train your model, push it on
        params = random.choice(param_choices)
        model = classtype(**params)

        if (dev):
            best = [0, 0, 0]
            subset_dev = [[{k:v for k,v in dev[0][i].iteritems() if k in feature_choices} for i in range(len(dev[0]))], dev[1]]
            for i,weights in enumerate(model.train_epochs(subset_train[0], subset_train[1], 2)):
                results = model.test(*subset_dev)
                accuracy = sum(r[0] == r[1] for r in results) / float(len(results))
                if accuracy > best[0]:
                    best[0] = accuracy
                    best[1] = weights
                    best[2] = i+1
            model = classtype(initial_weights=best[1], **params)
        else:
            model.train(*subset_train)

        models.append(model)
    
    print
    return models

def bag_transform(models, data):
    data[0] = [{i:model.predict(obsv) for i,model in enumerate(models, start=1)} for obsv in data[0]]
    for obsv in data[0]:
        obsv[0] = 1.0 # bias term

def bag_transform_all(models, train, test, dev, folds):
    bag_transform(models, train)
    bag_transform(models, test)
    bag_transform(models, dev)
    for f in folds:
        bag_transform(models, f)

