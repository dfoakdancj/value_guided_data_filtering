## Experiment: further ablation over the optimistic data collection

### Setup

####  Modify the source code of Gym
To load our xml file in **env/assets/**, we need to provide a variable for Walker2dEnv in the gym.

Go to **gym.envs.mujoco.${env(-v2)}**, and change the ${EnvClass}**.__init__()** from 

```
    def __init__(self):
```
to

```
    def __init__(self, xml_file="walker2d.xml"):
```

### Launch

Run the experiment w/o the optimistic data collection in Walker (5 runs with different seeds):


```
python train.py --env walker --no_optimistic --seeds 12 123 1234 12345 123456
```


Run the experiment w/wo optimistic data collection in Walker-morph (5 runs with different seeds):


```
python train.py --env walker_morph --no_optimistic --seeds 12 123 1234 12345 123456
```


### Results:

<div align="left">
  <img src="https://github.com/dfoakdancj/value_guided_data_filtering/blob/master/ablation_optimistic/LC_ablation_optimistic_wk.png?raw=true">
</div>



### Discussion

To validate the effect of optimistic data collection in more complex environments that require advanced exploration techniques, we further conducted experiments in Walker environments. The results demonstrate that the application of optimistic data collection techniques improves the performance of our proposed method in the Walker environments. It is worth noting that our proposed method is not designed specifically for solving sparse-reward settings. However, our method can be combined with existing exploration techniques to address these challenges.