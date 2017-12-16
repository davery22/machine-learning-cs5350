# Final project - twitter spambot filter
The culmination of my journey through the U's Machine Learning course. The goal is to learn to label twitter accounts as spam or not based on pre-aggregated features.

## Organization
- `data/` -- Contains the datasets and related information (some raw data, feature descriptions, etc.)
- `classifiers/` -- Contains (the best versions of) all of my classifiers throughout the course
- `ensemble/` -- Contains a couple additional classes for ensemble support (i.e. Bagger and AdaBoost)

- `featurizer.py` -- An experiment in collecting additional features from the raw data
- `trainer.py` -- Helpful methods like cross-validation, majority baseline, ensemble sub-model training, and input transformation
- `solver.py` -- The rest; data extraction, discretization, normalization, and all the experimental setup

Clearly not pefect. Documentation is lacking, files and methods are bulky, the object flow could use work, `trainer.py` and `solver.py` are in need of redistribution, it can't do everything you'd dream... But good enough for a one-time academic use?

## How to use
`> ./run.sh`

aka

`> python solver.py`

And thus it runs whatever experiments you have set up. Currently I have several potentials commented out (in `solver.py/main()`) in a way that should make sense. It's only a few lines difference to go from cross-validating various hyper-parameters of LogisticRegressor to running AdaBoost over Perceptrons trained on DecisionTree outputs based on discretized data.

