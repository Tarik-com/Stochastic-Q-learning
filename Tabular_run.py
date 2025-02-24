from Args import *
from functions import*
from agents import *
import pickle


args=Args_() 
models_name=["Random_model","Q_Learning_model","Stoch_Q_Learning_model","Double_Q_Learning_model","Stoch_Double_Q_Learning_model","Sarsa_model","Stoch_Sarsa_model"]
results={}
for name in models_name:
    results[f"rewards_{name}"]=[]
    results[f"acu_rewards_{name}"]=[]
    results[f"episodes_length_{name}"]=[]
for i in range(10):
    print(f"step: {i}")
    Random_model=RandomAgent(args)

    Q_Learning_model=QLearningAgent(args)
    Stoch_Q_Learning_model=Stoch_QLearningAgent(args)

    Double_Q_Learning_model=DoubleQLearningAgent(args)
    Stoch_Double_Q_Learning_model=Stoch_DoubleQLearningAgent(args)

    Sarsa_model=SARSAAgent(args)
    Stoch_Sarsa_model=Stoch_SARSAAgent(args)

    models=[Random_model,Q_Learning_model,Stoch_Q_Learning_model,Double_Q_Learning_model,Stoch_Double_Q_Learning_model,Sarsa_model,Stoch_Sarsa_model]
    
    for model,name in zip(models,models_name):
        reward,acu_reward,length=model.train()
        results[f"rewards_{name}"].append(reward)
        results[f"acu_rewards_{name}"].append(acu_reward)
        results[f"episodes_length_{name}"].append(length)
        

for name in models_name:
    results[f"average_rewards_{name}"]=np.average(np.array(results[f"rewards_{name}"]),axis=0)
    results[f"average_acu_rewards_{name}"]=np.average(np.array(results[f"acu_rewards_{name}"]),axis=0)
avg_rewards=[]
avg_acu=[]

for name in models_name:
    avg_rewards.append(results[f"average_rewards_{name}"])
    avg_acu.append(results[f"average_acu_rewards_{name}"])
results["average_rewards"]=avg_acu
results["average_acumulated_rewards"]=avg_acu

with open(f'/../../data/I6326771/FL_05.pkl', 'wb') as f:
    pickle.dump(results, f)