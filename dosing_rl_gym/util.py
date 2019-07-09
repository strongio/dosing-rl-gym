import numpy as np
import gym
import matplotlib.pyplot as plt

def plot_env(env_name, n_episodes):
    env = gym.make(env_name)
    env.set_episode_length(10)
    glucose_list = []
    action_list = []
    for e in range(n_episodes):
        env.reset()
        for _ in range(env.episode_length-2):
            action = [2.8]
            state, reward, done, _ = env.step(action)
            #if done:
                #break
        glucose_list.append(env.G)
        action_list.append(env.u)

    # Plot the results
    plt.figure(1, figsize=(11,8))
    ticks = np.linspace(0,24,13)

    ax = plt.subplot(3,1,1)
    plt.plot(env.t, env.u, 'b--', linewidth=3)
    plt.ylabel('Insulin Pump (mU/min)')
    plt.xlim([0,24])
    plt.xticks(ticks)
    plt.title(env_name)

    ax = plt.subplot(3,1,2)
    for l in glucose_list:
        plt.plot(env.t, l, 'r-',linewidth=2, alpha=0.2)
    plt.plot(env.t, np.array(glucose_list).mean(axis=0),'r-',linewidth=2, label='Mean Glucose')
    plt.plot([0,24],[105,105],'k--',linewidth=2,label='Reward bounds')
    plt.plot([0,24],[80,80],'k-',linewidth=2,label='Reward target')
    plt.plot([0,24],[65,65],'k--',linewidth=2)
    plt.ylabel('Glucose (mg/DL)')
    plt.legend(loc='best')
    plt.xlim([0,24])
    plt.xticks(ticks)
    plt.xlabel('Time (hour)')
    plt.show()