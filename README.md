# JeepneyPhaseoutSentimentPrediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Description

Uses Random Forests, Decision Tree, & Logistic Regression models to predict twitter posts are for or against the jeepney phaseout in the Philippines

Warning: Bias exists in the project and requires more raw data to train the models. 

## Main files and how to replicate

All datasets required can be found in the "data folder." The sentiments.csv file is the dataset that has the most up to date and cleaned data for this project. 
Some initial warnings: Some of the text data has not been perfectly translated to English. 

In order to run the models and see the model accuracy and results, refer to py/modeling and run each of the three phaseout files. These should output three different tables on their classification reports.
The logistic regression file will also display a confusion matrix for interpretability.
The decision tree model will also display the root node and it's branches.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources. (Data was collected by self)
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         py and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── py   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes py a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    |   └── phaseout_modeling_decisiontree.py      <- Decision Tree Model
    |   └── phaseout_modeling_logreg.py            <- Logistic Regression Model
    |   └── phaseout_modeling_randforest.py        <- Random Forest Model
    |
    │
    └── plots.py                <- Code to create visualizations
```

--------

