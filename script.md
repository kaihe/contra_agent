### Session 1
1. first thought of reward mechanism, use score directly. Experiment 1 to see the baseline. Discuss the reasonable reward value range, maybe a tanh function of score between (-10, 10), use 
2. shape the reward to make it a mixture of 



### Session 2
compare the max travel distance of following experiments, each trained with 10m steps:
1. state == original 
2. state == falling_trap
3. mixture of the two initial states