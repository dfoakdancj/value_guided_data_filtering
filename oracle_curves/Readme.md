## Oracle learning curves


<div align="left">
  <img src="https://github.com/dfoakdancj/value_guided_data_filtering/blob/master/oracle_curves/LC_ablation_oracle.png?raw=true">
</div>


### Discussion

We analyzed the sample size of the Oracle algorithm and plotted the learning curves of both Oracle and value_guided_data_filtering as a function of the target domain sample size. Our findings indicate that Oracle converges to near-optimal performance with 1e6 samples from the target domain in most environments. Additional 1e5 transitions may improve performance, but we believe that the marginal benefit of these additional transitions would not be significant.