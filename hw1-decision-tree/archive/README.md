# Decision Tree Experiments
## Code
Run all experiments sequentially:

`> ./run.sh` or `> python decision_tree.py`

Vocabulary:
- 'Feature' = an attribute of the data, with multiple potential values
- 'Feature value' = the value that a given feature takes in a single observation
- 'Observation' = list of feature values, one for each feature
- 'Feature index' = index at which you can find a given feature's value in an observation
- 'Observations' = list of observations
- 'Labels' = list of labels, indices correspond to indices of observations

The decision tree works as follows:
- The n-ary decision tree takes a list of observations, which is a list of lists of feature values. Thus the data must be ‘featurized’ before being processed by the tree. The tree also takes a list of labels. The observations and labels correspond by index.
- Tree nodes have a value and dictionary of children.
- There is no type distinction between a decision node and a label node. You can tell
you’re at a label node when the node has no children.
- For decision nodes, the value is the index of the feature in the original list of feature
values. Thus, feature indices must be consistent across observations, and missing
features must be resolved before training.
- For label nodes, the value is the label.
- The dictionary of children in a decision node is keyed using feature values.
- When predicting, if the observation contains an unseen feature value, it takes the path
of the first known feature value in the children dictionary. This allows the tree to still
make a prediction, and hopefully lower-order features will be relevant.
- During training, the tree optionally takes a maxDepth parameter (default = -1). Values
less than 1 result in an unconstrained-depth tree.
- During training, the tree optionally takes an error function (default = entropy;
majorityError is also implemented in the file).

## Experiment 1
This experiment uses an unconstrained-depth decision tree based on ID3. It reads in the train file and extracts the feature values for each observation, and also gets the labels for each observation. It trains the tree, reporting the depth achieved. Then it tests on the training data, reporting the training accuracy. Next it reads in the test file, again extracting features and labels. It tests on the test data and reports accuracy.

## Experiment 2
This experiment runs a 4-fold cross validation on trees with constrained depths of {1,2,3,4,5,10,15,20}. The features and labels for each fold are extracted from the CVSplits files. Cross-validation averages and standard deviations of the accuracy are calculated for each depth. Then a formula is used to evaluate the best-performing depth. Finally, experiment 1 is repeated using this depth limit for the tree.
