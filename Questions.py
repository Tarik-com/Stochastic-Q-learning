###1 how to set the size of mini batch and check for the size of buffer size and sample size
"""
when i sample from the buffer, the data is stored like [n_envs,dimension]
so when i sample a batch, the output is of the shape [batch,n_envs,dim]

so when i chose elements i chose pairs of n_envs elements. 
so i am forced to chose all the actions that are linked to the same n_envs



when i select the action i reshape the action sampled from [batch,n_env,action_dim] to [batch*n_env, action_dim]
and i end up with n_env*log(n) action

for the learning is it necessary to train the model with the action that comes 
only from the specific environment we are training or can we train one env with
the data from another

"""