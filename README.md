# CM50270_CW2
Reinforcement Learning CW2 - Comparing Approaches To Develop Agents To Learn To Play Super Mario Bros!

## Setting up the Python dependencies
This must be done prior to launching any of the notebooks / running any of the agents' code.
### Conda (requirements.txt)
If you're using conda, you can create an environment with the required dependencies using the `requirements.txt` by running the below:
```bash
conda create --name $ENV_NAME --file requirements.txt
```
This will create a conda environment called `$ENV_NAME`, which can be activated with the following:
```bash
conda activate $ENV_NAME
```
If you wish to use conda and do not already have it installed, please follow the installation instructions found here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

## Pipenv (Pipfile)
If you're using pip, you can create an environment with the required dependencies using the `Pipfile` by running the below:
```bash
pipenv install
```
If you do not currently have pipenv and wish to use it, please review the installation instructions found here: https://pipenv.pypa.io/en/latest/install/#installing-pipenv

## Running the Notebooks
```bash
# Navigate to notebook parent dir
cd src

# If using pipenv or conda envrionments, please load them now

# launch JupyterLab
jupyter-lab
```

Once in JupyterLab, navigate to the notebook you wish to execute (A2C agent, DQN Agent, or CNN Feature Visualisation) and begin executing cells.

