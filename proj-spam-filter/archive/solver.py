import trainer
from classifiers import *
from ensemble import *
import random
import math

def get_train_file():
    return './data/data-splits/data.train'

def get_test_file():
    return './data/data-splits/data.test'

def get_eval_file():
    return './data/data-splits/data.eval.anon'

def get_eval_id_file():
    return './data/data-splits/data.eval.id'

""" Discretization """

def frange(start, step, count):
    ret = []
    i = 0
    while i < count:
        ret.append(start + i*step)
        i += 1
    return ret

def get_discretizations(lis, maxi, mini, n):
    ret = []
    section_size = (float(maxi)-mini)/n
    magnets = frange(maxi, -section_size, n)
    for item in lis:
        # Sub in closest magnet
        ret.append(min(magnets, key=lambda x: abs(x-item)))
    return ret

def inner_discretize(data, max_cardinality):
    example_count = len(data[0])
    feature_length = max(map(max, data[0]))+1
    for f in range(1,feature_length): # ignore bias term
        # Get max and min for this feature
        maxi, mini = float('-inf'), float('inf')
        for example in data[0]:
            if example.get(f,0) > maxi:
                maxi = example.get(f,0)
            if example.get(f,0) < mini:
                mini = example.get(f,0)

        # Get discretization for each feature value
        discretizations = get_discretizations([ex[f] for ex in data[0]], maxi, mini, max_cardinality)
        
        # Reassign feature values to their discretizations
        for i,example in enumerate(data[0]):
            example[f] = discretizations[i]

    return data

def discretize(full_train, train, test, dev, folds, cardinality=5):
    # Combine all to discretize
    full_train_len = len(full_train[0])
    full_train[0] += test[0]
    full_train[1] += test[1]

    full_train = inner_discretize(full_train, cardinality)

    # Reinstate original splits
    train = [d[:len(train[0])] for d in full_train]
    dev = [d[len(train[0]):len(train[0])+len(dev[0])] for d in full_train]
    len_folds = [len(f[0]) for f in folds]
    folds = []
    for i,_ in enumerate(len_folds):
        if i == 0:
            folds.append([d[:len_folds[i]] for d in full_train])
        else:
            start = sum(len_folds[:i])
            folds.append([d[start:start+len_folds[i]] for d in full_train])
    test = [d[full_train_len:] for d in full_train]
    full_train = [d[:full_train_len] for d in full_train]

    return full_train, train, test, dev, folds


""" Normalization """

def normalize(data):
    keys = [k for k in data[0][0] if k > 0] #ignore bias term
    jlen = len(data[0])
    for k in keys:
        maxi,mini,mean = float('-inf'), float('inf'), 0
        for j in range(jlen):
            mean += data[0][j][k]
            if data[0][j][k] > maxi:
                maxi = data[0][j][k]
            if data[0][j][k] < mini:
                mini = data[0][j][k]
        mean /= float(jlen)
        stddev = 0
        for j in range(jlen):
            stddev += (data[0][j][k] - mean)**2
        stddev = (stddev/(jlen-1))**0.5
        for j in range(jlen):
            #data[0][j][k] = float(data[0][j][k] - mini)/(maxi - mini)       # [0,1] scaling
            #data[0][j][k] = 2*float(data[0][j][k] - mini)/(maxi - mini)-1   # [-1,1] scaling
            #data[0][j][k] = float(data[0][j][k] - mean)/(maxi - mini)       # mean normalization, good
            data[0][j][k] = float(data[0][j][k] - mean)/stddev              # standardization, real good
    return data

""" Data extraction """

def extract_data(infile):
    observations = []
    labels = []

    with open(infile, 'r') as infile:
        for line in infile.readlines():
            splits = line.strip().split(' ')
            labels.append(2*int(splits[0])-1)
            splits = splits[1:] if len(splits) > 1 else []
            features = {int(kv[0]): float(kv[1]) for kv in [val.split(':') for val in splits]}

            # Add log features
            i = 0
            length = max(features)+1
            for j in range(1,length): #[7,8,16]
                v = features[j]
                features[length+i] = math.log(v if v > 0 else -v) if v else -5.0
                i += 1

            features[0] = 1.0 # introduce bias term

            observations.append(features)

    return [observations, labels]

def get_all_data():
    full_train = normalize(extract_data(get_train_file()))
    test = normalize(extract_data(get_test_file()))
    train_test = 0.7
    folds = 5
    count = len(full_train[0]) # number of labels/observations
    train = [d[:int(count*train_test)] for d in full_train]
    dev = [d[int(count*train_test):] for d in full_train]
    t_count = len(full_train[0])/folds + 1
    folds = [[d[i:i+t_count] for d in full_train] for i in range(0, len(full_train[0]), t_count)]

    return full_train, train, test, dev, folds

def get_eval_data():
    full_train = extract_data(get_train_file())
    test = extract_data(get_test_file())
    full_train[0] += test[0]
    full_train[1] += test[1]
    full_train = normalize(full_train)
    test = normalize(extract_data(get_eval_file())) # overwrite test with eval

    train_test = 0.7
    folds = 5
    count = len(full_train[0]) # number of labels/observations
    train = [d[:int(count*train_test)] for d in full_train]
    dev = [d[int(count*train_test):] for d in full_train]
    t_count = len(full_train[0])/folds + 1
    folds = [[d[i:i+t_count] for d in full_train] for i in range(0, len(full_train[0]), t_count)]

    return full_train, train, test, dev, folds

""" Experiments """

def experiment1(classtype, train, test, dev, folds, param_choices={}):
    ''' for linear models that support training in epochs and cross-validation '''

    # Cross-validate to find best hyperparameters
    print 'Cross-validating to find best hyperparameters in', param_choices, '...'
    acc, params = trainer.select_hyperparams(classtype, folds, param_choices)
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

    return results

def experiment2(classtype, train, test, folds, param_choices):
    ''' for models that don't support epoch training '''

    # Cross-validate to find best hyperparameters
    print 'Cross-validating to find best hyperparameters in', param_choices, '...'
    acc, params = trainer.select_hyperparams(classtype, folds, param_choices)
    print '* Best Params:', params
    print '* Best Params Average Cross-val Accuracy: {0:.4f}'.format(acc)
    
    # Use best params to test
    learner = classtype(**params)
    learner.train(train[0], train[1])

    print 'Testing against \'train\''
    results = learner.test(train[0], train[1])
    accuracy = sum(r[0] == r[1] for r in results)
    print '* Train Accuracy: {0}/{1} ({2:.4f})'.format(accuracy, len(results), float(accuracy)/len(results))

    print 'Testing against \'test\''
    results = learner.test(test[0], test[1])
    accuracy = sum(r[0] == r[1] for r in results)
    print '* Test Accuracy: {0}/{1} ({2:.4f})'.format(accuracy, len(results), float(accuracy)/len(results))

    return results

def experiment3(classtype, train, test, params):
    ''' for models where cross-validation is infeasible (*cough* ensembles *cough*) '''

    # Train the learner
    learner = classtype(**params)
    learner.train(train[0], train[1])

    print 'Testing against \'train\''
    results = learner.test(train[0], train[1])
    accuracy = sum(r[0] == r[1] for r in results)
    print '* Train Accuracy: {0}/{1} ({2:.4f})'.format(accuracy, len(results), float(accuracy)/len(results))

    print 'Testing against \'test\''
    results = learner.test(test[0], test[1])
    accuracy = sum(r[0] == r[1] for r in results)
    print '* Test Accuracy: {0}/{1} ({2:.4f})'.format(accuracy, len(results), float(accuracy)/len(results))

    return results

def report_results(results):
    results = [(r[0]+1)/2 for r in results]

    # Match results to ID's
    with open(get_eval_id_file()) as idFile:
        results = [','.join(r) for r in zip(idFile.read().splitlines(), map(str, results))]

    with open('results.csv', 'w') as outFile:
        results = '\n'.join(['Id,Prediction'] + results)
        outFile.write(results)
    
def main():
    # Use RNG seed for consistency during comparison testing
    random.seed(7)

    # Get all the data
    print 'Fetching data...'
    full_train, train, test, dev, folds = get_all_data() # get_all_data() OR get_eval_data()
    #full_train, train, test, dev, folds = discretize(full_train, train, test, dev, folds, cardinality=9) # need to do all together, so trees aren't confused by new values

    # Go
    print '--- Majority Baseline ---\n'
    trainer.majority_baseline(test, dev)

    print '\n--- Experiment ---\n'
    # Non-ensemble runs
    #results = experiment1(perceptron.Perceptron, train, test, dev, folds, param_choices={'learning_rate': [1,0.1,0.001],'margin':[0.0,1.0],'aggressive':[True,False],'dynamic_learning':[True,False]})
    #results = experiment1(svm.SVM, train, test, dev, folds, param_choices={'learning_rate': [10, 0.1, 0.0001], 'tradeoff': [1000, 100, 10]})
    results = experiment1(logisticregressor.LogisticRegressor, train, test, dev, folds, param_choices={'learning_rate': [10,1,0.01], 'sigma_squared': [1000,float('inf')]})
    #results = experiment2(naivebayes.NaiveBayes, full_train, test, folds, param_choices={'smoothing':[2.0,1.5,1.0,0.5,0.0]})
    #results = experiment2(decisiontree.DecisionTree, train, test, folds, param_choices={'depth':[3,5,7,9], 'error':[decisiontree.majorityError, decisiontree.entropy]})

    # Collect models for Bagging/Stacking experiments
    #models = trainer.get_trained_models(perceptron.Perceptron, train, dev, {'learning_rate':[1,0.1,0.001], 'margin':[0,0.1,1]}, 10, 100)
    #models = trainer.get_trained_models(svm.SVM, train, dev, {'learning_rate':[1,0,0.001,0.00001], 'tradeoff':[1000, 10, 1]}, 15, 10)
    #models = trainer.get_trained_models(logisticregressor.LogisticRegressor, train, dev, {'learning_rate':[1,0.1,0.01], 'sigma_squared':[float('inf')]}, 20)
    #models = trainer.get_trained_models(naivebayes.NaiveBayes, full_train, None, {'smoothing':[0.0]}, 100, 1000, 16)
    #models = trainer.get_trained_models(decisiontree.DecisionTree, full_train, None, {'depth':[7]}, 100, 2000, 16)

    # Bagged models
    #results = experiment3(bagger.Bagger, full_train, test, {'classifiers':models})

    # Stacked models
    #trainer.bag_transform_all(models, train, test, dev, folds)
    #results = experiment2(decisiontree.DecisionTree, train, test, folds, {'depth':[3,10,15,20], 'error':[decisiontree.entropy, decisiontree.majorityError]})
    #results = experiment1(logisticregressor.LogisticRegressor, train, test, dev, folds, param_choices={'learning_rate': [10,1,0.1], 'sigma_squared': [100, 10000, float('inf')]})

    # AdaBoost
    #results = experiment3(adaboost.AdaBoost, train, test, params={'iters':50, 'dev':dev, 'epochs':20, 'classtype':perceptron.Perceptron, 'params':{'learning_rate':0.001, 'margin':1.0}})

    # *Use for eval runs
    #report_results(results)

if __name__ == '__main__':
    main()
