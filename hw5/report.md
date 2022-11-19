# CS285 HW5 Report

## Part 1 “Unsupervised” RND and exploration performance

Performance Compare
||Eval Average|
|--|--|
|Easy|![](images/q1_easy.png)|
|Medium|![](images/q1_med.png)|

State Density Comparison

||Random|RND|
|--|--|--|
|Easy|![](images/q1_easy_random_curr_state_density.png)|![](images/q1_easy_rnd_curr_state_density.png)|
|Medium|![](images/q1_med_rand_curr_state_density.png)|![](images/q1_med_rnd_curr_state_density.png)|

### Subpart Custom Exploration: Boltzman

![](images/q1_alg_boltz.png)

# Part 2 Offline learning on exploration data

## Subpart 1: Q value comparison
![](images/q2_qvals.png)

## Subpart 2 Numsteps comparison: 
![](images/q2_num_steps_compare.png)

## Subpart 3: Alpha comparison:
![](images/q2_alpha.png)

Alpha 0.2 performs the best while dqn performs the worst.

<br>

## Part 3 “Supervised” exploration with mixed reward bonuses.

### Compare to Q2(purely offline)
![](images/q3_p1.png)
Clearly, mixed reward is the winner.

### Compare to Q1(rnd with default exploration=10000steps)
![](images/q3_p2.png)
Even though the final result is close, but clearly CQL with mixed reward converges a lot faster than standard RND.

## Part 4 Offline Learning with AWAC 

||Supervised|Unsupervised|
|--|--|--|
|Easy|![](images/q4_ez_sup.png)|![](images//q4_ez_uns.png)|
|Medium|![](images/q4_med_sup.png)|![](images/q4_med_us.png)|

Best lambda: Easy-sup(10), Easy-unsup(10), Med-sup(2), Med_unsup(0.1)


## Part 5 Offline Learning with IQL

||Supervised|Unsupervised|
|--|--|--|
|Easy|![](images/q5_ez_sup.png)|![](images/q5_ez_unsup.png)|
|Medium|![](images/q5_med_sup.png)|![](images/q5_med_unsup.png)|

Best tau: Easy-sup(0.99), Easy-unsup(0.8), Med-sup(0.9), Med_unsup(0.9)

### Final compare CQL, AWAC, IQL
![](images/final_compare.png)

<br/>

From the plot, we can see that cql seems to performs the best in the end. AWAC is also really close and converges fast. IQL seems to perform the worst among all.