###1 how to set the size of mini batch and check for the size of buffer size and sample size
"""
# questions about the buffer and format 
when i sample from the buffer, the data is stored like [n_envs,dimension]
so when i sample a batch, the output is of the shape [batch,n_envs,dim]


so when i chose elements i chose pairs of n_envs elements. 
so i am forced to chose all the actions that are linked to the same n_envs



when i select the action i reshape the action sampled from [batch,n_env,action_dim] to [batch*n_env, action_dim]
and i end up with n_env*log(n) action

for the learning is it necessary to train the model with the action that comes 
only from the specific environment we are training or can we train one env with
the data from another

# question about the stoch dqn target value.
when we sample from the buffer and evaluate the target value for the sampled obs, 
do we evaluate those based on the actions sampled from the same sample or do we sample new actions?

# discretizing the action space
should i first discretize tha action space and then after use the env wwith those discretized actions 
or should i use the env with continious actions and then discretiz those actions
"""