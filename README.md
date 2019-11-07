# twentyoneflow
Cosmological 21cm signal modelling with tensorflow

## Version

Version: 1.0

## Overview

`twentyoneflow` allows on to make fast predictions of the 21cm signal using deep neural networks developed and trained using `tensorflow`. The

An example of using pre-trained models to make predictions can be found in `notebooks/example.ipynb`.

NOTE: All training of models in this repo was done using the 21cm cosomology training data found at: 

https://zenodo.org/record/3532141#.XcSg7NHLc5k

Furthermore, `example.ipynb` uses the 21cm cosmology test data available at: 

https://zenodo.org/record/3530920#.XcRlKjP7QuU

These two datasets were obtained from randomly subsampling a dataset which forms their union.

The pre-trained networks can be found in `saved_models/keras/`, while the preprocessing scalers are in `saved_models/scalers/`.

An example script for training your own model using `tensorflow/keras` can be found in `scripts/keras_model_template.py`. 
Similarly a template for `sklearn` networks can be found in `scripts/sklearn_model_template.py`.


## Requirements

This package was developed and tested using `python 2.7`, but should be compatible with newer versions such as `python 3.6`.

## Installation

### Pip install from PyPi

Coming soon.

### git install

No installation of `twentyoneflow` is strictly necessary, one can simply clone the repo:

`git clone https://github.com/SuperKam91/twentyoneflow.git`,

and run their code from there, or add the location of the directory to their PYTHONPATH environment variable. 
Note however that one must ensure that all the required packages are installed by running `pip install -r requirements.txt` from
the root the directory.

Alternatively, one can run 

`python setup.py install` 

or 

`pip install .` 

from the root of the repo to install the base requirements (i.e. those in `requirements.txt`) as well as the `twentyoneflow` package.

If the previous method doesn't work, first run `python setup.py bdist_wheel` followed by `python -m pip install dist/<wheel file name>.whl` where `<wheel file name>` will depend on the version of `python` used when running `setup.py` and the version number contained within the file.

## Contributing to the `twentyoneflow` package

Want to contribute to `twentyoneflow`? There are two main you can contribute via the [GitHub repository](https://github.com/SuperKam91/twentyoneflow):

### Opening issues

Open an issue to report bugs or to propose new features.

### Proposing pull requests

Pull requests are very welcome. Note that if you are going to propose drastic changes, please open an issue for discussion first, to make sure that your proposed change will be accepted before you spend effort implementing it.
