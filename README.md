# DRL for Testing a Software System or Game
### CSCI 4440U Adv. Web Dev 

### Ryan Don, Siddhant Das, 2025

All files for respective games are located in /mario/ and /doom/

Check out the README sections for each game:

- [Super Mario Bros - NES](#super-mario---nes)

- [Doom](#doom)

---

# Super Mario - NES

<div style="text-align: center;">
    <img style="width: 45%;" src="mario/notebooks/resources/marios.gif" alt="Several mario instances training" />
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
This may take quite a while, stable-retro and stable_baselines3 are especially big packages
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
cd /mario
python3 -m src.eval_mario
```

### References

We used the maintained fork of Gym Retro, Stable Retro, for our experiments [Poliquin, 2025].

Poliquin, M. (2025). *Stable Retro, a maintained fork of OpenAI's gym-retro*. GitHub repository. Retrieved from https://github.com/Farama-Foundation/stable-retro

---

# Doom

<div style="text-align: center;">
    <img style="width: 45%;" src="" alt="Doom Training/Eval GIF" />

    TODO: Make gif of doom
</div>

## Background

This uses <a href="https://github.com/Farama-Foundation/ViZDoom">ViZDoom</a>, a lightweight environment based on the classic first-person shooter DOOM, allowing agents to interact with the game through raw visual input.

The installation includes several built-in scenarios, each defined by configuration files specifying maps, objectives, and accessible game variables (health, ammo, kills, etc). Adding new scenarios simply involves placing their .wad and .cfg files in the scenarios directory, after which they can be loaded as standard Gymnasium environments.

## How to Setup

TODO: Create setup instructions


## How to train & evaluate

TODO: Create train & eval instructions

### References

We used ViZDoom for our visual reinforcement learning experiements [Wydmuch et al., 2019].

Wydmuch, M., Kempka, M., & Jaśkowski, W. (2019). ViZDoom Competitions: Playing Doom from Pixels. IEEE Transactions on Games, 11(3), 248–259. https://doi.org/10.1109/TG.2018.2877047