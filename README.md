# AD2C - Adaptive Diversity Control

This repository introduces **AD2C (Adaptive Diversity Control)**, a framework for enhancing **multi-agent reinforcement learning (MARL)** by dynamically managing behavioral diversity. It extends the work of **[DiCo: Controlling Behavioral Diversity in Multi-Agent Reinforcement Learning (Bettini et al., ICML 2024)](https://openreview.net/forum?id=qQjUgItPq4)** by introducing a novel adaptive control mechanism based on Extremum Seeking Control (ESC). This allows AD2C to intelligently balance exploration and exploitation to solve complex, heterogeneous MARL tasks like multi-agent navigation.

<p align="center">
<img src="https://github.com/Svar7769/AD2C-Apdaptive-Diveresity-Control/blob/Master/es_controller/AD2C/assets/ES%20Controller.png" alt="ES Controller FlowChart">
</p>

---

## 🔹 How It Works: The ES Controller

The core of this project is the **HetControlMlpEsc** model, which uses an **Extremum Seeking Control (ESC)** loop to dynamically tune the level of **behavioral diversity** among agents.  

Unlike fixed-diversity methods, AD2C treats diversity as a parameter to be optimized **in real-time** based on the team's performance (reward).

## The Key Components are:
- **Shared Policy (`shared_mlp`)**: A base policy shared by all agents.
- **Agent-Specific Policy (`agent_mlps`)**: Individual Policies that create Diverse behaviours.
- **ESC Loop**:
  1. A sinusoidal **dither signal** is added to the diversity scaling factor (`k_hat`).
  2. The perturbed factor scales the outputs of agent-specific policies.
  3. The system observes the **team reward (J)**.
  4. Reward is **demodulated** with the dither to estimate the gradient of reward w.r.t. diversity.
  5. `k_hat` is updated via **gradient ascent** to maximize team reward.

 This allows the system to continuously "feel out" the **optimal level of diversity** at any given time, adapting to the evolving dynamics of the task.

---

## 🛠️ Installation

Run This using **Python 3.9** in a virtual environment (e.g., conda).

```bash
# Create environment
conda create -n ad2c python=3.9
conda activate ad2c
````

### 1. Install dependencies (forks with `het_control` branch)

```bash
# VMAS
git clone -b het_control https://github.com/proroklab/VectorizedMultiAgentSimulator.git
pip install -e VectorizedMultiAgentSimulator

# TensorDict
git clone -b het_control https://github.com/matteobettini/tensordict.git
cd tensordict && python setup.py develop && cd ..

# TorchRL fork
git clone -b het_control https://github.com/matteobettini/rl.git
cd rl && python setup.py develop && cd ..

# BenchMARL
git clone -b het_control https://github.com/matteobettini/BenchMARL.git
pip install -e BenchMARL
```

### 2. (Optional) Install logging tools

```bash
pip install wandb moviepy
```

### 3. Install AD2C

```bash
git clone https://github.com/Svar7769/AD2C-Apdaptive-Diveresity-Control.git
pip install -e AD2C-Apdaptive-Diveresity-Control
```

---

## 🚀 Running Experiments

All experiment entry points are located in `ad2c/run_scripts/`.

### Run a single experiment

```bash
python ad2c/run_scripts/your_run_script.py model.your_parameter=0.3
```

### Run across multiple seeds or parameter grids

```bash
python ad2c/run_scripts/your_run_script.py -m model.your_parameter=0.1,0.5,0.9 seed=0,1,2
```

### Example with multiple overrides

```bash
python ad2c/run_scripts/your_run_script.py \
    model.your_parameter=0.3 \
    seed=1 \
    experiment.max_n_frames=2_000_000 \
    algorithm.lmbda=0.9
```

---

## ⚙️ Configuration

* All experiment configs are in: `ad2c/conf/`
* Structure follows **BenchMARL** (algorithm, model, task, experiment hyperparameters)
* **Hydra** lets you combine and override configs at runtime

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

* [ ] Add Proper Visulaizations for the inner working of the algorithm
* [ ] Benchmark results for AD2C vs DiCo 
---

## 🙌 Acknowledgements

This repository builds upon:

* [DiCo](https://openreview.net/forum?id=qQjUgItPq4)
* [BenchMARL](https://github.com/matteobettini/BenchMARL)
* [TorchRL](https://github.com/pytorch/rl)
* [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator)

---
