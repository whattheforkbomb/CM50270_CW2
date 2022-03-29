# CM50270_CW2
Reinforcement Learning CW2 - Comparing Approaches To Develop Agents To Learn To Play Super Mario Bros!

## Installing Clang
Please refer to the below documentation to install Clang:
- Linux (Debian) - https://github.com/Kautenja/nes-py#debian
- Windows - https://github.com/Kautenja/nes-py#windows
## Setting up the Python dependencies (conda)
If you're using conda, you can create an environment with the required dependencies using the `src/conda/rl-conda_dependencies.yml` by running the below:
```bash
conda env create -f src/conda/rl-conda_dependencies.yml
```
This will create a conda environment called `rl`, which can be activated with the following:
```bash
conda activate rl
```
Alternatively you can manually install the required python dependencies using your Python package manager of choice, either using `src/conda/rl-conda_dependencies.yml` as a guide, or by following the instructions found on the below sites:
- https://github.com/Kautenja/nes-py#readme - NES emulator (required to run Super Mario Bros)
- https://pypi.org/project/gym-super-mario-bros/ - The Super Mario Bros OpenAI Gym environment
- https://jupyter.org/install - Install Jupyter lab to run the notebook

## Launching the notebook
```bash
# Navigate to notebook parent dir
cd src

# if using conda
conda activate rl

# launch JupyterLab
jupyter-lab
```

Once in JupyterLab, navigate to the notebook and begin executing cells.

