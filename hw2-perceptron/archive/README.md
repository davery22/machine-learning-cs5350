# Perceptron Experiments
## Code
Run all experiments sequentially:

`> ./run.sh` or `> python perceptron.py`

High level design notes:
- The perceptron uses mixed sparse and dense representations of data. In particular, the internal weight vector(s) are encoded as 'dense' arrays, while the observations are encoded as 'sparse' dictionaries (mapping feature indices to values).
- The above allows for interesting benefits. Namely, potentially faster dot products and sums (if the observation is truly sparse), and the ability to incorporate more/new features in the observations without restarting training. The cost is that the weight vector's width needs to be checked (and maybe grown) for each observation, which involves finding the max index of the observation features.
- Thus, my perceptron's input demands deviate from those of my previously designed decision tree. The underlying design decision was to prioritize capabilities of each model over interchangability of models (contrast to something like scikit-learn). 

Particular design notes:
- The Perceptron class encapsulates every variant of the perceptron that the experiments consider. Different variants are configured via different keyword arguments to the constructor. Keywords include:
    - `learning_rate` [default=1] - Optionally specifies a (initial) learning rate.
    - `dynamic_learning` [default=False] - Optionally forces the learning rate to decay each time an update is applied.
    - `margin` [default=0] - Optionally specifies a marginal offset affecting the update check.
    - `aggressive` [default=False] - Optionally forces updates to use an aggressive learning rate. If specified this option overrides `learning_rate` and `dynamic_learning`.
    - `average` [default=False] - Optionally specifies to track an average weight vector, used for predictions.
    - `initial_weights` [default=[]] - Optionally specifies to intitialize the weight vector (and average weight vector) with the provided weights.
- The Perceptron class includes a batch `train()` and a `train_online()`. Thus, it is compatible in both domains. `train()` is actually a wrapper around `train_online()`, and they can be safely used together if convenient.
- The Perceptron class also includes a `train_epochs()` generator, convenient for shuffling data and extracting weights after each epoch.

## Experiments
Each experiment has the same form. First, run 5-fold cross validation for every combination of hyperparameters, 10 epochs per fold, and obtain the best-performing hyperparameter combination. Next, train the classifier for 20 epochs (with the previous hyperparameters), and evaluate the resultant weights from each epoch against the dev set. Finally, choose the weights from the best epoch and evaluate against the test set.

Experiments, in order:
1. Simple Perceptron
2. Perceptron with Dynamic Learning Rate
3. Margin Perceptron
4. Averaged Perceptron
5. Aggressive Perceptron with Margin

## Notes
It takes about 2 minutes to run through all 5 tests. Have time for a snack?
