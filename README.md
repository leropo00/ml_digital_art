# Digital Art AI agents and ML training experiments


An experiments projects, with AI agents and ML training notebooks,
for the domain of digital art creation.
This is still project in the beginning, so many changes will likely happen.

## Project setup

To set up your environment variables, you need to duplicate the `.env.example` file and rename it to `.env`. You can do this manually or using the following terminal command:

```bash
cp .env.example .env # Linux, macOS, Git Bash, WSL
copy .env.example .env # Windows Command Prompt
```

This command creates a copy of `.env.example` and names it `.env`, allowing you to configure your environment variables specific to your setup.

Currenlty the following environment variables are needed:

```
OPENAI_API_KEY=key provided by openai
```


It is recommended to run the project inside a python virtual environment.
One of the ways to create virtual environment is the following:

```
# be in the project root folder
python -m venv ./.venv

# how to activate the environment
. .venv/bin/activate

# when iniside virtual environment, you leave with command
deactivate

# install the dependencies, present in uv.lock
uv sync

# 
uv pip install -e .
```

The uv package manager is used, instead of more common pip.


## Project commands

Inside the file:   `pyproject.toml`, are multiple defined commands

When adding a new command to pyproject.toml
use `uv pip install -e .` 
so tha command will be visible

## Project Organization

The project structure was created from this template:  https://github.com/datalumina/datalumina-project-template.
Most of the folder are currently empty, the structure of the project may change.

Currently the following folders are of interest:

```
├──  database,  contains the files, that are used for creating data models, by sqlc.
├──  notebooks, contains some Jupyter Notebookes. I personally created notebooks at google collab. 
├	 Currently some notebooks fail in github viewer, but they work if you download them and view them separately.
├──  project, contains all the files needed by project that are runnable.
    │
    ├──  app             <- Contains the code by the fastai server
    │
    ├──  src             <- Contains the implementation of services, that perform actions.
	│
    ├──  test            <- Contains the code samples, that can always be run via commandline
    │
├── requirements.txt   <- The requirements file for reproducing the environment,
							generated with `pip freeze > requirements.txt`
```

The project structure was created from this template:  https://github.com/datalumina/datalumina-project-template.

This is the original datalumina project structure

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py 
```

--------