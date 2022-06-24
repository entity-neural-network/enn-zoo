# Entity Neural Network Zoo

[![Actions Status](https://github.com/entity-neural-network/enn-zoo/workflows/Checks/badge.svg)](https://github.com/entity-neural-network/enn-zoo/actions)
[![Discord](https://img.shields.io/discord/913497968701747270?style=flat-square)](https://discord.gg/SjVqhSW4Qf)


The enn-zoo package collects [entity-gym](https://github.com/entity-neural-network/entity-gym) bindings for a number of reinforcement learning environments:
- [Procgen](https://github.com/openai/procgen)
- [Griddly](https://github.com/Bam4d/Griddly)
- [MicroRTS](https://github.com/santiontanon/microrts)
- [ViZDoom](https://github.com/mwydmuch/ViZDoom)
- [CodeCraft](https://github.com/cswinter/DeepCodeCraft)

## Setup

```
git clone https://github.com/entity-neural-network/enn-zoo.git
cd enn-zoo
poetry install
poetry run pip install setuptools==59.5.0
poetry run pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

Some of the environments have additional dependencies:

```
sudo apt install python3-dev make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

## Usage

```
poetry run python enn_zoo/train.py
```

### Gym-µRTS

[Gym-µRTS](https://github.com/vwxyzjn/gym-microrts) is a Reinforcement Learning environment for the popular Real-time Strategy game simulator μRTS. To get started, run the following command:

```python
xvfb-run -a poetry run python enn_zoo/train.py \
    env.id=GymMicrorts \
    rollout.num_envs=24 \
    total_timesteps=1000000 \
    rollout.steps=256 \
    track=true \
    eval.capture_videos=True \
    eval.interval=300000 \
    eval.steps=2000 \
    eval.num_envs=1
```

Here is a [tracked Gym-µRTS experiment](https://wandb.ai/entity-neural-network/enn-ppo/runs/1vpdd0cm?workspace=user-costa-huang), which has a trained agent that behaves as follows:

https://user-images.githubusercontent.com/5555347/175696804-e151a790-5324-45f1-9f2f-6f1f885a6e35.mp4
