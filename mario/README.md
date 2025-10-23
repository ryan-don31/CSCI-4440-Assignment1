- How to setup
- How to train and evaluate
- Environments (actions, observations, rewards)
- Reproduction steps for the reported results (what exact values did we test with for the screenshots and pics)
- GIFs for each app, screenshots as well

## How to Setup

Required versions

python3.8 - 3.10
Ubuntu22.04 (Not required, strongly recommended)
<a href="https://apps.microsoft.com/detail/9pn20msr04dw?hl=en-US&gl=CA">Ubuntu 22.04.5 LTS WSL On Microsoft Store</a>

Create virtual environment
python -m venv venv
source ./venv/bin/activate

Install requirements
pip install -r requirements.txt 
This may take quite a while, stable-retro and stable_baselines3 are especially big packages


(OPTIONAL) Install packages used for rendering in training
sudo apt update
sudo apt-get install libgtk-3-dev


Add a Mario ROM
Download the Super Mario Bros (World).nes ROM through LEGAL and NON-PIRACY MEANS
(Please don't come after me nintendo ninjas)

cd "/path/to/rom/installation/"
python3 -m retro.import .


## How to train & evaluate

