## Experiment: data-filtering based on the dynamics discrepancies

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

Walker (5 runs with different seeds):

```
python train.py --env walker --seeds 12 123 1234 12345 123456
```


Walker-morph (5 runs with different seeds):

```
python train.py --env walker_morph --seeds 12 123 1234 12345 123456
```

HalfCheetah (5 runs with different seeds):

```
python train.py --env halfcheetah --seeds 12 123 1234 12345 123456
```

HalfCheetah-morph (5 runs with different seeds):

```
python train.py --env halfcheetah_morph --seeds 12 123 1234 12345 123456
```

Amt (5 runs with different seeds):

```
python train.py --env ant --seeds 12 123 1234 12345 123456
```

Ant_morph (5 runs with different seeds):

```
python train.py --env ant_morph --seeds 12 123 1234 12345 123456
```

Hopper (5 runs with different seeds):

```
python train.py --env hopper --seeds 12 123 1234 12345 123456
```

Hopper_morph (5 runs with different seeds):

```
python train.py --env hopper_morph --seeds 12 123 1234 12345 123456
```

### Results:

<div align="left">
  <img src="https://github.com/dfoakdancj/value_guided_data_filtering/blob/master/dynamics_gap_driven_data_filtering/LC_ablation_dynamics_equal_selection.png?raw=true">
</div>



### Discussion

We compared the value_guided_data_filtering algorithm with a variant that incorporates data-filtering based on the dynamics gap. The likelihood of the source domain transitions on the target domain dynamics model determines the dynamics gap. The results show that the variant underperforms value_guided_data_filtering in most environments.