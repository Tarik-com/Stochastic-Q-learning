from Args import *
from functions import*
from agents import *
import pickle
import time
start_time = time.time()


args=Args_()
models_name=["DQN"]
print(f" total steps: {args.total_timesteps} env: {args.env_id} number of env {args.num_envs}  models names: {models_name}")


results={}
results["reward_per_episode"]=[]
results["actions"]=[]
results["Q_Values"]=[]
    
for i in range(2):
    f = open("DQN_IP_run.txt", "a")
    f.write(f"step: {i} time: {time.time() - start_time}")
    f.close()
    
    DQN=DQNAgent(args)

    models=[DQN]
    
    for model,name in zip(models,models_name):
        reward_per_episode,actions,q_values=model.train()
        results["reward_per_episode"].append(reward_per_episode)
        results["actions"].append(actions)
        results["Q_Values"].append(q_values)

results["info"]=[(time.time() - start_time)]
with open('DQN_IP_rewards.pkl', 'wb') as f:
    pickle.dump(results["reward_per_episode"], f)
with open('DQN_IP_actions.pkl', 'wb') as f:
    pickle.dump(results["actions"], f)
with open('DQN_IP_qvalues.pkl', 'wb') as f:
    pickle.dump(results["Q_Values"], f)
