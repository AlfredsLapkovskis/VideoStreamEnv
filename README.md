# Source code for the paper "Benchmarking Dynamic SLO Compliance in Distributed Computing Continuum Systems"

## Alfreds Lapkovskis, Boris Sedlak, Sindri Magnússon, Schahram Dustdar & Praveen Kumar Donta (2025)

##### Preprint: https://arxiv.org/abs/2503.03274

##### Code for the sample client iOS app: https://github.com/AlfredsLapkovskis/SmartVideoStream


# 1. Setup

We used Python 3.11.5.

## 1.1. Configure a Virtual Environment

Execute these commands from the project root directory:

**Windows**:
```batch
python -m venv env
env/bin/activate
pip install -r requirements.txt
```

**Linux**:
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**MacOS:**
```zsh
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## 1.2. Create Config

Copy example_config.json into the project root directory. Name it as **config.json**.

Specify there:

- **resource_dir:** path to the directory (relative to the root) where different resources (e.g., metrics, models, plots) will be stored.
- **logs_dir:** path to the directory (relative to the root) where logs with performance measurements (SLO compliance, CPU and memory usage) will be stored.


## 1.3. Unzip the Dataset

To simulate connection and communication between a client and the server, unzip **metrics_data.zip** to get pre-collected datasets with metrics from a client: 
- **metrics.csv** contains metrics collected under regular conditions. 
- **metrics_rate_limit.csv** contains metrics collected under limited bandwidth.

This dataset is necessary for running the reinformcement learning agent, even in real-time environment, because it is used to compute parameters (mean and std) to standardize real-time client metrics.


## 1.4. Prepare Sample Videos

To actually run the server that can stream videos to clients, you need to prepare sample videos, otherwise, meaningful experiments can only be conducted using pre-collected metrics (see 1.3). For this, you need to run this command from the root directory on each of your videos:

```zsh
sh gen_jpegs.sh <video file name>
```
_(Note, tested on MacOS only, for Windows, etc. you may need some other utility instead of 'sh' or some extra steps)_

This will generate jpegs from your video of different resolutions and store them at **videos** directory in the project root.

# 2. Project Structure

- **experiments** a directory containing JSON files with experiment hyperparameters and settings. These files are parsed by **experiment.py** and must be named as **exp<i>\<index></i>.json**.
- **learning** a directory containing classes related to different agents that aim to learn optimal streaming configurations to maximize SLO compliance:
    - **agent.py**, **base_agent.py** and **base_hyperparameters.py** contain base classes for agents and hyperparameters.
    - **active_inference** directory contains our implementation of the Active Inference agent based on [Sedlak et al., 2024](https://www.sciencedirect.com/science/article/pii/S0167739X24002887) and their [source code](https://github.com/borissedlak/workload/tree/main/FGCS) with our modifications (**active_inference.py**), and a data class with hyperparameters (**hyperparams.py**).
    - **reinformcement_learning** directory contains our implementation of a reinforcement learning agent that supports DQN, A2C, and PPO algorithms (**reinforcement_learning_agent.py**), its hyperparameters (**hyperparams.py**), definition of an environment (**environment.py**), and some other utilities.
    - **random** and **sequential** directories represent some dummy agents that generate random streaming configurations or iterate them sequentially.
- **models** directory contains some useful data models used across the project:
    - **messages.py** defines and implements models of messages exchanged via WebSockets.
    - **metrics.py** defines a model with metrics that we obtain from a client.
    - **service_level_objectives.py** defines service level objectives with which a client's streaming configuration should comply (with the help of an agent).
    - **session.py** stores context of a client session.
    - **slo_fulfillment.py** stores individual SLO compliance scores, and provides methods to compute overall SLO compliance, and related statistics.
    - **stream_settings.py** defines a streaming configuration.
- **services** contain some classes that implement communication with clients via WebSockets:
    - **message_receiver.py** handles messages received from a client.
    - **message_sender.py** provides methods to send messages to a client.
    - **message_sender_with_rate_limit.py** same as **message_sender.py** but allows to simulate a limited bandwidth.
    - **video_streamer.py** implements video streaming.
- **utils** directory contains various utilities used in the project:
    - **data_iterators** directory contains entities that a/synchronously iterate a dataset with pre-collected metrics or metrics received from a client in real time.
    - **logging** directory contains classes used for logging agent performance metrics.
    - **preprocessing** directory contains classes that preprocess raw client metrics into suitable formats used by agents. Also, **heating_metric_preprocessor_wrapper.py** specifically allows to wrap a metric preprocessor to simulate device heating/cooling based on its workload.
    - **experiment.py** a class containing hyperparameters and settings for multimodal model training. It parses JSON experiment definitions from the **experiments** directory.
    - **frame_loader.py** contains a class that loads video frames for streaming. It uses multiple threads and prefetches frames to reduce latency.
    - **message_coder.py** encodes and decodes messages sent or received from WebSockets.
    - **usage_monitor.py** helps to measure CPU and memory usage.
    - **common.py** contains some common utilities, and **variables.py** defines constants and names of variables that we work with across the project. 
- **train.py** is a script to run an experiment using pre-collected metrics (requires a JSON with experiment definition; see above about these experiments).
- **server.py** is a script used to launch the server (also requires a JSON with experiment definition; see above about these experiments).
- **tensorboard_to_csv.py** a script to convert tensorflow log files to CSVs with performance metrics.
- **experiment_stats.ipynb** a notebook that shows some statistics about feasible SLO compliance rates across different experiments.
- **plot_results.ipynb** a notebook that we used to plot our results.
- **logs.zip** a zip with CSVs with our results.
- **metrics_data.zip** a zip with pre-collected datasets of metrics:
    - **metrics.csv** contains metrics collected under regular conditions. 
    - **metrics_rate_limit.csv** contains metrics collected under limited bandwidth.


# 3. Execute Code

Execute all the code from the root project directory. Please check source files for expected parameters. For example, to run an experiment with configuration **exp1.json** and pre-collected metrics, you could execute:

```zsh

python -m train -e 1 -d metrics_data/metrics.csv -s True

```

Or, to launch a server with **exp1.json**, you could use:

```zsh

python -m server -e 1 -d metrics_data/metrics.csv

```
_(see 1.3. for why the server still may need '-d metrics_data/metrics.csv' specified)_.
