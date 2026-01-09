# AD2C - Adaptive Diversity Control

This repository introduces **AD2C (Adaptive Diversity Control)**, a framework for enhancing **multi-agent reinforcement learning (MARL)** by dynamically managing behavioral diversity. It extends the work of **[DiCo: Controlling Behavioral Diversity in Multi-Agent Reinforcement Learning (Bettini et al., ICML 2024)](https://openreview.net/forum?id=qQjUgItPq4)** by introducing a novel adaptive control mechanism based on Extremum Seeking Control (ESC). This allows AD2C to intelligently balance exploration and exploitation to solve complex, heterogeneous MARL tasks like multi-agent navigation.

<p align="center">
<img src="https://github.com/Svar7769/AD2C/blob/main/src/AD2C_framework.png" alt="ES Controller FlowChart">
</p>

---

## 🔹 How It Works: The ES Controller

The core of this project is the **HetControlMlpEsc** model, which uses an **Extremum Seeking Control (ESC)** loop to dynamically tune the level of **behavioral diversity** among agents.  

Unlike fixed-diversity methods, AD2C treats diversity as a parameter to be optimized **in real-time** based on the team's performance (reward).

### Key Components:
- **Shared Policy (`shared_mlp`)**: A base policy shared by all agents.
- **Agent-Specific Policy (`agent_mlps`)**: Individual policies that create diverse behaviors.
- **ESC Loop**:
  1. A sinusoidal **dither signal** is added to the diversity scaling factor (`k_hat`).
  2. The perturbed factor scales the outputs of agent-specific policies.
  3. The system observes the **team reward (J)**.
  4. Reward is **demodulated** with the dither to estimate the gradient of reward w.r.t. diversity.
  5. `k_hat` is updated via **gradient ascent** to maximize team reward.

This allows the system to continuously "feel out" the **optimal level of diversity** at any given time, adapting to the evolving dynamics of the task.

---

## 🛠️ Installation

### Prerequisites
- Linux (tested on Ubuntu)
- CUDA 11.7 (for GPU support)
- Python 3.9

### Step-by-Step Setup

#### 1. Install Miniconda

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run installation script
bash Miniconda3-latest-Linux-x86_64.sh

# Apply changes to current shell
source ~/.bashrc

# Activate conda (adjust path if installed elsewhere)
source ~/miniconda3/bin/activate
```

#### 2. Create and Activate Environment

```bash
# Create environment with Python 3.9
conda create --name ad2c python=3.9

# Activate the environment
conda activate ad2c
```

#### 3. Install Dependencies

Install the required forks with custom branches:

**VMAS (Vectorized Multi-Agent Simulator):**
```bash
git clone -b topic/env_setup https://github.com/Svar7769/VectorizedMultiAgentSimulator.git
pip install -e VectorizedMultiAgentSimulator
```

**TensorDict:**
```bash
git clone -b het_control https://github.com/Svar7769/tensordict.git
cd tensordict && python setup.py develop && cd ..
```

**TorchRL:**
```bash
git clone -b het_control https://github.com/Svar7769/rl.git
cd rl && python setup.py develop && cd ..
```

**BenchMARL:**
```bash
git clone -b task/env_setup https://github.com/Svar7769/BenchMARL.git
pip install -e BenchMARL
```

#### 4. Install Logging Tools

```bash
pip install wandb moviepy
```

#### 5. Install AD2C

```bash
git clone -b stabel https://github.com/Svar7769/AD2C-Diversity-Testing.git
cd AD2C-Diversity-Testing
pip install -e .
```

#### 6. Install PyTorch 1.13.1 (Required Version)

```bash
# Remove any existing PyTorch installation
conda remove pytorch torchvision torchaudio
pip uninstall torch torchvision torchaudio

# Install PyTorch 1.13.1 with CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

#### 7. Install Additional Dependencies

```bash
pip install matplotlib plotly
```

#### 8. Fix NumPy Version

```bash
# Ensure NumPy version < 2 for compatibility
pip uninstall numpy
pip install "numpy<2"
```

---

## 🚀 Running Experiments

AD2C uses a modular runner system where all common functionality is centralized in `run.py`, and task-specific parameters are defined in individual runner scripts.

### File Structure

```
het_control/
├── conf/                              # Configuration files
│   ├── balance_ippo_config.yaml      # Hydra config for balance task
│   ├── navigation_ippo.yaml          # Hydra config for navigation task
│   └── callback/
│       └── escontroller.yaml         # ESC controller parameters
│
└── run_tasks/                        # Runner scripts
    ├── run.py                        # Common runner (handles all tasks)
    ├── run_balance.py                # Balance task runner
    ├── run_navigation.py             # Navigation task runner
    └── run_simple_tag.py             # Simple tag task runner
```

### Running a Task

Navigate to the `run_tasks` directory and execute your desired task:

```bash
cd het_control/run_tasks/

# Run balance task with ESC
python run_balance.py

# Run navigation task without ESC
python run_navigation.py

# Run simple tag with adversarial ESC
python run_simple_tag.py
```

### Creating a New Task

Create a new runner script `run_newtask.py`:

```python
from run import run_experiment

# Paths
ABS_CONFIG_PATH = "/path/to/het_control/conf"
CONFIG_NAME = "newtask_config"
SAVE_PATH = "/path/to/checkpoints/newtask/"

# Training parameters
MAX_FRAMES = 1_200_000
CHECKPOINT_INTERVAL = 500_000
DESIRED_SND = 0.0

# Task-specific overrides
TASK_OVERRIDES = {
    "n_agents": 4,
    # Add other task parameters
}

# ESC Controller
USE_ESC = True
ESC_CONFIG_FILE = "/path/to/het_control/conf/callback/escontroller.yaml"

if __name__ == "__main__":
    run_experiment(
        config_path=ABS_CONFIG_PATH,
        config_name=CONFIG_NAME,
        save_path=SAVE_PATH,
        max_frames=MAX_FRAMES,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        desired_snd=DESIRED_SND,
        task_overrides=TASK_OVERRIDES,
        esc_config_path=ESC_CONFIG_FILE,
        use_esc=USE_ESC
    )
```

### Overriding ESC Parameters

You can override ESC parameters directly in your runner script without creating a new config file:

```python
# In run_simple_tag.py
ESC_OVERRIDES = {
    "control_group": "adversary",  # Control adversary instead of agents
    "dither_magnitude": 0.3,
    "max_snd": 4.0,
}
```

---

## ⚙️ Configuration

### ESC Controller Configuration

Edit `het_control/conf/callback/escontroller.yaml` to tune ESC parameters:

```yaml
esc_controller:
  control_group: "agents"        # "agents" or "adversary"
  initial_snd: 0.0               # Starting diversity value
  dither_magnitude: 0.2          # Exploration amplitude
  dither_frequency: 1.0          # Exploration frequency (rad/s)
  integrator_gain: -0.001        # Controller gain
  high_pass_cutoff: 0.5          # High-pass filter (rad/s)
  low_pass_cutoff: 0.1           # Low-pass filter (rad/s)
  use_adaptive_gain: true        # Enable adaptive gain
  sampling_period: 1.0           # Sampling period (seconds)
  min_snd: 0.0                   # Minimum SND value
  max_snd: 3.0                   # Maximum SND value
  use_action_loss: false         # Enable action space loss
  action_loss_lr: 0.001          # Action loss learning rate
```

### Task Configuration

All task-specific configurations follow BenchMARL's structure:
- **Algorithm config**: Learning rate, batch size, etc.
- **Model config**: Network architecture, activation functions
- **Task config**: Environment parameters (n_agents, rewards, etc.)
- **Experiment config**: Total frames, evaluation frequency, logging

Configurations are in `het_control/conf/` and can be overridden via command-line arguments.

---

## 📊 Experiment Organization

### Enabling/Disabling ESC

In any runner script, simply set:

```python
USE_ESC = True   # Enable ESC controller
# or
USE_ESC = False  # Disable ESC (use fixed diversity)
```

### Checkpoint Management

Checkpoints are automatically saved based on your configuration:

```python
SAVE_PATH = "/path/to/checkpoints/task_name/"
CHECKPOINT_INTERVAL = 500_000  # Save every 500k frames
```

### Logging with Weights & Biases

If you have `wandb` installed, logging is enabled by default. Configure in your Hydra config:

```yaml
experiment:
  logger: wandb
  project_name: "ad2c-experiments"
```

---

## 🔬 Example Experiments

### Balance Task with ESC
```bash
cd het_control/run_tasks/
python run_balance.py
```
Trains agents to balance collaboratively while ESC optimizes behavioral diversity.

### Navigation Task (Fixed Diversity)
```bash
python run_navigation.py
```
Runs multi-agent navigation with fixed diversity settings (no ESC).

### Simple Tag (Adversarial ESC)
```bash
python run_simple_tag.py
```
Competitive task where ESC controls the adversary group's diversity.

---

## 🐛 Troubleshooting

### CUDA/PyTorch Issues

If you encounter CUDA errors:
```bash
# Check CUDA version
nvidia-smi

# Reinstall correct PyTorch version
pip uninstall torch torchvision torchaudio
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

### NumPy Compatibility

If you see NumPy-related errors:
```bash
pip uninstall numpy
pip install "numpy<2"
```

### Module Not Found

Ensure all packages are installed in editable mode:
```bash
cd VectorizedMultiAgentSimulator && pip install -e .
cd ../tensordict && python setup.py develop
cd ../rl && python setup.py develop
cd ../BenchMARL && pip install -e .
cd ../AD2C-Diversity-Testing && pip install -e .
```

### ESC Config File Not Found

Ensure `escontroller.yaml` exists at:
```
het_control/conf/callback/escontroller.yaml
```

Or update the path in your runner script:
```python
ESC_CONFIG_FILE = "/absolute/path/to/escontroller.yaml"
```

---

## 📖 Citation

If you use this repository, please cite the original DiCo paper:

```bibtex
@inproceedings{bettini2024controlling,
    title={Controlling Behavioral Diversity in Multi-Agent Reinforcement Learning},
    author={Bettini, Matteo and Kortvelesy, Ryan and Prorok, Amanda},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=qQjUgItPq4}
}
```

---

## 📌 Roadmap

- [ ] Add proper visualizations for the inner workings of the algorithm
- [ ] Benchmark results for AD2C vs DiCo across multiple tasks
- [ ] Support for additional MARL environments
- [ ] Hyperparameter tuning guidelines and best practices
- [ ] Pre-trained model checkpoints for quick evaluation

---

## 🙌 Acknowledgements

This repository builds upon:

* [DiCo](https://openreview.net/forum?id=qQjUgItPq4) - Original diversity control framework
* [BenchMARL](https://github.com/matteobettini/BenchMARL) - Multi-agent benchmarking library
* [TorchRL](https://github.com/pytorch/rl) - Reinforcement learning framework
* [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) - Vectorized multi-agent simulator

Special thanks to the ProrokLab team for their foundational work on behavioral diversity in MARL.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

## 📄 License

This project is licensed under the same terms as the original DiCo repository. Please refer to the LICENSE file for details.
