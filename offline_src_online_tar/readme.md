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
  <img src="https://github.com/dfoakdancj/value_guided_data_filtering/blob/master/offline_src_online_tar/LC_offline_oracle.png">
</div>



### Discussion

In order to verify the efficacy of our approach in the scenario involving an offline source domain dataset and an online target domain simulator, we have leveraged the CQL algorithm as our backbone algorithm and have executed experiments based on the d4rl offline datasets. Across all environments, we have utilized medium datasets and conducted 10 gradient updates for each online interaction with the target domain. Our results demonstrate that VGDF can effectively operate within the offline-online context, and we contend that further improvements in dataset quality and the number of gradient updates would yield even better outcomes.