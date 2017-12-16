# Homework 5: Plagiarism Detection
Here we experiment with 6 different classifiers, training on ~2800 examples with ~58000 binary features each, to predict whether a speech was plagiarized.

## Code
Run all experiments sequentially:

`> ./run.sh` or `> python trainer.py`

But since you don't want to wait ~2 hours, I'd suggest just looking at `results.txt`.

### High level design notes:
- Data files are referenced via hard-coded strings in `get_all_data()` in `trainer.py`. This method also handles splitting the train file into train and dev splits (for classifiers that train in epochs; currently an 80-20 split).
- As a mild cleanliness step, each classifier is broken into its own file. Full-featured classifiers (having `train`, `test`, and `predict` methods -- all of them except `Bagger) inherit from `Learner`.
- For now all the cross-validation and experimental setup is nestled in `trainer.py`.

### Classifier design notes:
SVM
- SVM is remarkably similar to my previous Perceptron implementation (which also vestigially resides in this directory). The key difference is the Hinge Loss update, implying sub-gradient descent. If the prediction is in the margin, or on the wrong side, an additional scaled loss is applied to the new weights; otherwise the weights are merely scaled (approaching 1) based on a decaying learning rate. Hyper-parameters include `learning_rate` and `tradeoff`.

Logistic Regression
- Also built on top of the Perceptron shell, the Logistic Regressor merely uses yet another loss function (derived in the written homework) to update the weights. This time the hyper-parameters are `learning_rate` and `sigma_squared`. I diverged from my pseudo-code implementation in the writeup, in favor of faster element-wise computations instead of mapping operations. The example shuffling is also done externally, so the classifier is not worried about handling epochs.

Naive Bayes
- A classifier formed by counting examples to generate a prior (probabilities of each label) and likelihoods (probabilities of each feature taking a value, given a label). For predictions, the probablities of each feature appearing in the given example are multiplied with the prior for a label, and the label with the greatest product (posterior probability) is chosen. I probably would have gotten better results using log likelihoods (to avoid limits of floating point precision rounding to 0, especially with so many features in the product), but I didn't this time.

Decision Tree
- Never used directly in the experiments, but I did renege on my prior implementation in favor of sparse observations. Partly to save space; mainly so I didn't have to reformat the data between experiments.

Bagger
- This class isn't what I'd call a "full" classifier, because you can't train with it. But it does take on the role of a voting classifier for testing, by giving it a list of trained classifiers to each make predictions, and then siding with the majority.

### Experimental setup:
As mentioned up top, there are 6 experiments we are running. They each follow a similar form, but adjusted where necessary to match the behavior of the classifier being tested. In `trainer.py` these adjustments break out into 3 separate `experiment#()` functions, but they are only different in terms of cross-validation, training in epochs, etc.

Experiments 1 & 2 use SVM and Logistic Regressor on the original dataset, cross-validating with several combinations of hyper-parameters (36 total). The best-performing set of parameters are chosen for epoch training, and then the best linear classifier from the epochs is evaluated on the train and test sets.

Experiment 3 uses the Naive Bayes classifier on the original dataset, cross-validating to pick the best smoothing hyper-parameter, and then testing against train and test.

At this point, 1000 decision trees are trained to a depth of 3 on the original dataset, each using 100 examples drawn I.I.D from the train set.

Experiment 4 bags those decision trees into a voting classifier and measures its accuracy against train and test.

Now, the trees are used to construct a feature transformation on all the data sets. The new features of each observation are the predictions of trees 1..1000 for the original observation.

Experiments 5 & 6 repeat the SVM and Logistic Regressor experiments, but using the transformed features.

### Other notes and caveats:
- Every classifier that has to run cross-validation has a `train_epochs()` method -- for those that are not linear classifiers, they just ignore the iterations and pass the input to the normal `train()` method to be processed once.
- SVM and Logistic Regressor still have some unused parameters leftover from Perceptron. The important ones are `learning_rate` (i.e. initial_learning_rate), `tradeoff` (SVM) or `sigma_squared` (Logistic Regressor), `dynamic_learning` (defaults to `True` for learning rate decay), and I guess `initial_weights` for epoch testing.
- SVM and Logistic Regressor of course use a decaying learning rate. Let `r0 = initial learning rate`; `rt = current learning rate`; `t = update count`. While I initially used `rt = r0 / (1 + t)`, I found that this decayed too quickly and both classifiers had horrible accuracy (worse than random choice). Currently the code implements `rt = r0 / (1 + (r0 * t) / 100)`, based off a suggestion in the course slides [svm-sgd slide 63]. That `100` is obviously a magic constant worthy of being its own hyper-parameter, but I figured there were enough hyper-parameters to go around in the experiments. 
- In these implementations, both SVM and Logistic Regressor only update the weight vector on elements present in the example. While not true to the exact math, this was really the only way to keep runtimes... reasonable? Most of the point of having sparse representations would be defeated if every update called for 58000 elements to be recomputed. It would also require a fixed-size weight vector based on upfront knowledge of feature count (which I explicitly avoided in my Perceptron implementation). So, I know it's not perfect, but if you really want to sit on your hands you can change a couple lines of code and wait a loooong time.
- The Naive Bayes Classifier in `naive_bayes.py` assumes binary features, for the sake of computational speedup in these experiments. A more general-purpose Naive Bayes was in-progress in `naive_bayes2.py`, but not intended for use yet.
