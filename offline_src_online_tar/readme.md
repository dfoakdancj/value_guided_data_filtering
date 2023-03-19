## Experiment: Offline source domain dataset + Online target domain interactions

### Setup

####  Modify the source code of Gym
To load our xml file in **env/assets/**, we need to provide a variable for Walker2dEnv in the gym.

Go to **gym.envs.mujoco.${env(-v2)}**, and change the ${EnvClass}**.__init__()** from 

```
    def __init__(self):
```

to

```
    def __init__(self, xml_file="halfcheetah/walker2d/ant/hopper.xml"):
```

### Launch

Run the experiment with d4rl halfcheetah-medium dataset in HalfCheetah (5 runs with different seeds):


```
python train.py --env halfcheetah --seeds 12 123 1234 12345 123456
```

Run the experiment with d4rl halfcheetah-medium dataset in HalfCheetah-morph (5 runs with different seeds):


```
python train.py --env halfcheetah_morph --seeds 12 123 1234 12345 123456
```

Run the experiment with d4rl hopper-medium dataset in Hopper (5 runs with different seeds):


```
python train.py --env hopper --seeds 12 123 1234 12345 123456
```

Run the experiment with d4rl hopper-medium dataset in Hopper-morph (5 runs with different seeds):


```
python train.py --env hopper_morph --seeds 12 123 1234 12345 123456
```


### Results:

<div align="left">
  <img src="https://github.com/dfoakdancj/value_guided_data_filtering/blob/master/offline_src_online_tar/LC_offline.png">
</div>



### Discussion

To validate the effect of our method in the setting with the source domain offline dataset and the target domain online simulator, we use CQL as our backbone algorithm and conduct experiments based on the d4rl offline datasets. We use medium datasets for all environments and perform 10 gradient updates per online interaction with the target domain. 