###Dependencies:

python                             3.7.1 <br/>
numpy                              1.15.4 <br/>
gpflow                             1.5.1 <br/>
tensorflow                         1.13.1 <br/>
gym                                0.15.4 <br/>
mujoco-py (optional)                          2.0.2.9 <br/>


### Build
1. pilco to be build as per README_PILCO.md <br/>
2. execute below and make sure it's working
```python
python sda.py -o true
```

### Usage
`mc_sda_pilco/carla/carla_controller.py` is the Carla client that are wired up with
synchronised MQ communication to ensure each step of rendering in Carla is only done
when `action` is received from `sda client`.

1. use `mc_sda_pilco/carla/carla_controller.py` to replace `testController.py` in Carla
examples and start `carla_controller.py`. This starts the MQ server and put Carla clients
on hold for the first `action` after sending out the initial `state`
2. start `mc_sda/carla/sda.py` in online mode (without giving any argument). Or start
`mc_sda/carla/test_client.py` which sends randomly generated actions for a fixed number
of times





