# TDT4173-machine-learning

This repository contains Keras models for weather prediction. 

## Table of Contents
- [Structure](#structure)
- [Installation](#installation)
- [Usage](#usage)


## Structure
The structure of the project. All the machine learning code is in the src folder.
```
├── data            # The data of the project
├── experiments     # Experiments
├── plots           # Plots of the data
├── src         
|   ├── data        # Processing data methods
|   ├── models      # Model architectures
|   ├── utils       # Different utils
|   ├── config.py   # Config file 
|   ├── evaluate.py # Evaluation of models
|   ├── main.py     # Main file for running the experiments
|   ├── requirements.txt
|   ├── train.py    # Training script
|   └── tuner.py    # Script for tuning the models 
├── tuning_results  # The tuning result appear here after tuning
└── README.md
```

The data directory contains the datasets used for this project. It contains the raw data, processed data, and data split into train, validation, and test sets. The experiments directory contains the results for each trained model after running the experiments. The plot directory contains vizualizations of the features in the dataset, as well as a heatmap showing the correleation between the feaetures. 

## Installation
To install the repository, run 
```
$ git clone https://github.com/kamidtli/tdt4173-machine-learning.git
```

### Requirements
- TensorFlow >= 2.31

To install the requirements, run
```
$ pip install -r requirements.txt
```
or if using Anaconda, run
```
$  conda create --name <env> --file requirements_conda.txt
```

## Usage 
To reproduce the experimental result, run the experiments using

```
$ python main.py
```

All the experimental parameters are being set in the config.py file. 
Here you can decide the learning rate, layers of the models, loss function, epochs, batch size, sequence length and data features.  
In the config file, you can also set if you want to train one model, or run the whole experiment by changing the experiments variable.
The structure of the experiments can also be set in the config file. You can both add more models in the models dictionary, or different feature combinations. 

To tune the models, run
```
$ python tuner.py
```
After running the tuner, the results appear in the tuning_results.txt file in the tuning_results folder.


### Reproducibility
To ensure reproducibility for the experiments, it is used a random seed, such that random functions always give the same results.
To remove this, remove this code from the main.py file
```python
# Set seed for reproducibility
randomState = 14
np.random.seed(randomState)
tf.random.set_seed(randomState)
``` 


 

 
