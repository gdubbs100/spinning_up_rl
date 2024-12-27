import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from gymnasium.wrappers import RecordVideo

def plot_training_results(batch_results):

    """
    Takes a dictionary of batch and training results and plots
    """

    results = {
        key: [batch[key] for batch in batch_results.values()] 
        for key in next(iter(batch_results.values()))
    }

    ## TODO: ensure these dimensions match the input batch
    fig, ax = plt.subplots(2,3)
    ax = ax.flatten()
    for i, res in enumerate(results.keys()):
        ax[i].plot(
            list(batch_results.keys()),
            results[res]
        )
        ax[i].set_title(res)
    plt.tight_layout()
    plt.show();

## TODO: add type hints for agent
def eval_agent(
    agent, 
    env: gym.Env, 
    num_eval_episodes:int, 
    save_video:bool=False, 
    video_folder:str ='./'
    ) -> dict[int: float]:

    """
    Runs a set of evaluation episodes and reports the achieved reward
    Optional recording of videos
    """

    if save_video:
        env = RecordVideo(
            env,
            video_folder ="./", 
            name_prefix="eval", 
            episode_trigger=lambda x: True
        )

    rewards = dict()

    for episode in range(num_eval_episodes):
        state, info = env.reset()
        rewards[episode]=0
        done = False
        while not done:

            with torch.no_grad():
                action, _ = agent.act(state)
            
            state, reward, terminated, truncated, info = env.step(action.numpy())
            done=terminated or truncated
            rewards[episode] += reward

    return rewards