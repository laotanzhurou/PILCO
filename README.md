###Disclaimer
Our original work in this program including source code and data files are all under `mc_sda_pilco`.

Files under `examples, pilco, tests` are of the credit of Nikitas Rontsis and Kyriakos Polymenakos, who authored the Python
implementation of PILCO. Our work in this project builds on top of the fork of their GitHub repository: https://github.com/nrontsis/PILCO

Ke Quan, 12 June 2020

###Dependencies:

python                             3.7.1 <br/>
numpy                              1.15.4 <br/>
gpflow                             1.5.1 <br/>
tensorflow                         1.13.1 <br/>
pyzmq                              17.1.2 <br/>
gym                                0.15.4 <br/>
mujoco-py (optional)                          2.0.2.9 <br/>


### Build
1. make sure above dependencies are in place.
2. pilco to be build as per README_PILCO.md <br/>

### File Structure
`mc_sda_pilco/carla/carla_controller.py` is the CARLA client resides in CARLA space. It is wired up with
synchronised MQ communication to ensure each step of rendering in CARLA is only done
when `action` is received from `sda client`.

`mc_sda_pilco/carla/test_client.py` is a script embeded with Data Gathering Policy - a random policy that either
increase or decrease `SunAltitude` in each step, with a bias towards decrease.

`mc_sda_pilco/carla_client.py` is the CARLA interface defined in Agent space, invoked by the agent implementation `sda.py`.

`mc_sda_pilco/sda.py` implements the reinforcement learning agent that described in the report.

`mc_sda_pilco/pilco_gp.py` is a wrapper of PILCO's MGPR package, which essentially constructs multiple GPs.

`mc_sda_pilco/environment.py` abstracts the CARLA environment for agent's internal operation, mainly for planning. Reward
function is defined in this class.

`mc_sda_pilco/mct.py` is our implementation of rho-UCT.

`mc_sda_pilco/util.py` collects multiple util function that we used for file I/O, conversion of data types etc.

Other than source code listed above, following folder in `mc_sda_pilco` stores run-time or design-time files that are important to the 
application:

1. `data/models` stores a list of pre-trained dumps of PILCO GPs. When working in online planning mode (`-m 3`) the program
will load a model dump as specified by `-l model_dump_name` and use that for planning.

2. `data/training_set` contains `state.txt` and `action.txt` which contains 200 episodes of data (each episode has about 50 state-action pairs)
that could be used for offline model learning

3. `data/test_set` contains a smaller set of data (about 20 episodes) for verification the prediction accuracy of model trained
using training data

4. `logs, figures and output` are folder where runtime information are dumped into. It's worth noting that model parameters learned during
offline model learnnig (`-m 2`) are dumped into `output` folder.


### Introduction
The main class or entry point of our program is `sda.py` which accepts a list of arguments:

`-m` the mode which application shall run. Theres are essentially 3 modes supported in  our program: 
1. `-m 1` runs a benchmark test, which loads a pre-trained model as specified by `-l model_dump_name` and performs a batch
of prediction accuray test against test data

2. `-m 2` is the model for offline model learning. At high level, it works in batches and for each batch it loads a set
of data from training files and optimise the GPs parameters. Model dumps are created and write to `output` folder after
completion of each batch.

3. `-m 3` is the mode for online planning, which load a model dump from `data/models` and initialise a rho-UCT as defined
in `mct.py` to interact with the CARLA environment in decision-time.

### Data Gathering
To run the data gathering policy, we need to copy the carla_controller.py to `Carla_dev/Carla 0.9.6/Python API/examples`.
Make sure the line 384 is uncommented, which gives randomised sun altitude for better coverage on model learning.
Once these are done follow below steps to gather training/test data:
1. launch the Carla server
2. in a separate terminal, activate conda environment CAL_latest, go to `Carla_dev/Carla 0.9.6/Python API/examples` then `python carla_controller.py`
3. in `mc_sda_pilco/carla` run `python test_client.py`

Change the hyper parameters defined in `python test_client.py` for number of episodes.

### Offline Model Learning
Once we have the training data and test data ready under `mc_sda_pilco/data` we could run the Offline Model Learning
mode to train up a PILCO GPs model.
```
python sda.py -m 2 -r 50 -b 10 -t 5 -d true
``` 
Execute above will tell our agent to run offline model learning by loading `-r 50` episodes from training data, break them down
into `5` batches given batch size `-b 10` and after each batch, we test the prediction accuracy by compare our one-step prediction results
with `-t 5` test sets. `-d true` specifies that our agent should dump the learned model parameters after completion of each batch. These
files will be dumped into `output` folder.

### Planning
To perform decision time planning, the procedure is similar to Data Gathering. But we need to make sure `carla_controller.py` configures the same
starting weather parameters on restart, check line 384 for mode details.
1. launch the Carla server
2. in a separate terminal, activate conda environment CAL_latest, go to `Carla_dev/Carla 0.9.6/Python API/examples` then `python carla_controller.py`
3. in `mc_sda_piloc` run `python sda.py -m 3 -l model_dump_name`

As our connection with `carla_controller` is anytime, for start a new planning we simply just need to rerun `sda.py` it will send message to
`carla_controller` to reset the environment.

### Running Test
For purely testing the prediction accuracy of learned models, use:
1. in another terminal, go to `mc_sda_pilco` make sure you have all the dependencies for our program, execute `python sda.py -m 1 -l name_of_model_dump -t 5`



