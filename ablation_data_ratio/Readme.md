## Experiment: different source domain transition size

### Setup

####  Modify the source code of Gym
To load our xml file in **env/assets/**, we need to provide a variable for HalfCheetah in the gym.

Go to **gym.envs.mujoco.${env(-v2)}**, and change the ${EnvClass}**.__init__()** from 

```
    def __init__(self):
```
to

```
    def __init__(self, xml_file="halfcheetah.xml"):
```

### Launch

Run the experiment with different transition ratios in HalfCheetah (5 runs with different seeds):


```
python train.py --env halfcheetah --tar_env_interact_freq 5/10/20 --seeds 12 123 1234 12345 123456
```


Run the experiment with different transition ratios in HalfCheetah-morph (5 runs with different seeds):


```
python train.py --env halfcheetah-morph --tar_env_interact_freq 5/10/20 --seeds 12 123 1234 12345 123456
```


### Results:

<div align="left">
  <img src="https://github.com/dfoakdancj/value_guided_data_filtering/blob/master/ablation_data_ratio/LC_ablation_tar_inter_freq.png?raw=true">
</div>


### Discussion

The results show that having more source domain transitions is beneficial when the number of target domain transitions is the same. This validates the ability of value_guided_data_filtering to fully leverage the source domain transitions and suggests that it is a promising approach for reinforcement learning in scenarios where data from the source domain is abundant.