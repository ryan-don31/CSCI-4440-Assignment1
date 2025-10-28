# DRL for Testing a Software System or Game

### Ryan Don, Siddhant Das, 2025

All files for respective games are located in /mario/ and /doom/

Check out the README sections for each game:

- [Super Mario Bros - NES](#super-mario---nes)

- [Doom](#doom)

Go check out the notebooks analysis for both games, as well:
- Mario: <a href="./mario/notebooks/mario.ipynb">mario.ipynb</a>
- Doom: <a href="./doom/notebooks/doom_analysis.ipynb">doom_analysis.ipynb</a>
---

# Super Mario - NES

<div style="text-align: center;">
    <img src="mario/notebooks/resources/gifs/marios.gif" alt="Several mario instances training" />

    Pictured: Training PPO mario speedrunner model with headless=False
</div>

## Background

This uses <a href="https://github.com/Farama-Foundation/stable-retro">stable-retro</a>, a maintained fork of OpenAi's gym-retro which allows classic video game ROMs to be wrapped in a gymnasium environment

the installation of stable retro contains directories for hundreds of games, each with files containing definitions for different in-game attributes (coins, lives, etc) tied
to their memory address in the ROM. importing a game to stable-retro simply drops the ROM into its directory, where it can then be used in a simple gymnasium environment.

## How to Setup

Required versions

- Python3.8 - 3.10
- Ubuntu22.04 (Not exactly required, but it's what I used)
<a href="https://apps.microsoft.com/detail/9pn20msr04dw?hl=en-US&gl=CA">Ubuntu 22.04.5 LTS WSL On Microsoft Store</a>


#### Create a virtual environment
```Shell
python -m venv venv
source ./venv/bin/activate
```

#### Install requirements
```Shell
pip install -r requirements.txt 
# This may take quite a while, stable-retro and stable_baselines3 are especially big packages
```

#### (OPTIONAL) Install packages used for rendering in training
```Shell
sudo apt update
sudo apt-get install libgtk-3-dev
```

#### Import the Super Mario Bros ROM
```Shell
# Download the Super Mario Bros (World).nes ROM through LEGAL and NON-PIRACY MEANS
# (Please don't come after me nintendo ninjas)

cd "/path/to/rom/installation/"
python3 -m retro.import .
```


## How to train & evaluate

The training and eval scripts each contain config values that I recommend you take a look at. This is where you can decide which algorithms, personas, etc to run.
- <a href="mario/src/eval_mario.py">eval_mario.py</a>
- <a href="mario/src/train_mario.py">train_mario.py</a>

#### Training
```Shell
cd /mario/
# Warning: do not cd into /src/ to run, if you are in any directory other than /mario/ (the current directory), the models/environments will not load correctly

python3 -m src.train_mario
```

Evaluating
```Shell
cd /mario/
python3 -m src.eval_mario
```

### References

We used the maintained fork of Gym Retro, Stable Retro, for our experiments [Poliquin, 2025].

Poliquin, M. (2025). *Stable Retro, a maintained fork of OpenAI's gym-retro*. GitHub repository. Retrieved from https://github.com/Farama-Foundation/stable-retro

---

# Doom

<div style="text-align: center;">
    <img src="doom/notebooks/resources/gifs/ppo_deadly_corridor.gif" style="width: 45%;" src="" alt="Instance of DRL agent running in Doom" />
    
    PPO model agent running during the deadly corridor scenario.
</div>
Note: Gifs of all models in all scenarios can be found in `doom/notebooks/resources/gifs/` and analysis is provided in `doom/notebooks/doom_analysis.ipynb`

## Background

This uses <a href="https://github.com/Farama-Foundation/ViZDoom">ViZDoom</a>, a lightweight environment based on the classic first-person shooter DOOM, allowing agents to interact with the game through raw visual input.

The installation includes several built-in scenarios, each defined by configuration files specifying maps, objectives, and accessible game variables (health, ammo, kills, etc). Adding new scenarios simply involves placing their .wad and .cfg files in the scenarios directory, after which they can be loaded as standard Gymnasium environments.

## How to Setup

To setup vizdoom:

```pip install vizdoom```

Additional packages in `doom/requirements.txt`

```
cd doom
pip install -r requirements.txt
```

## How to train & evaluate

### Training

Training is done within jupyter notebooks. To simply start training the models, run the cells from start to finish. More specifically, run the cell with `model.learn(...)` to start the training. The notebooks are seperated by scenarios and they are:

-   <a href="./doom/src/train_deadly_corridor.ipynb">train_deadly_corridor.ipynb</a>
-   <a href="./doom/src/train_defend_center.ipynb">train_defend_center.ipynb</a>

The environments used by these training files is found in: <a href="./doom/envs/doom_env.py">doom_env.py</a>

### Evaluation

To evaluate the models run <a href="./doom/src/eval_doom.py">eval_doom.py</a>:

```
cd doom/
python ./eval_doom.py
```

The evaluation program uses <a href="./doom/configs/eval.yaml">eval.yaml</a> as a configuration file for choosing the model and scenario. The possible values for the YAML configuration file are:
-   `algo`
    - `"ppo"` for Proximal Policy Optimization model 
    - `"a2c"` for Advantage Actor Critic model
-   `ppo`
    - `"defend_the_center"` for Defend the Center scenario
    - `"deadly_corridor` for Deadly Corridor scenario

### References

We used ViZDoom for our visual reinforcement learning experiements [Wydmuch et al., 2019].

Wydmuch, M., Kempka, M., & Jaśkowski, W. (2019). ViZDoom Competitions: Playing Doom from Pixels. IEEE Transactions on Games, 11(3), 248–259. https://doi.org/10.1109/TG.2018.2877047