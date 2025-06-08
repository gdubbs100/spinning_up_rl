import torch
import pandas as pd
import gymnasium as gym

from environments.continual_environment import ContinualEnv
from utils.training_utils import set_seed
from utils.logger import Logger


## TODO:
## - add tensorboard logging
## - add ability to save videos of performance
## - add ability to save model checkpoints
## - add task-inference option
## - review the steps counters, make sure this is working with the environment
## - clean up logics which are specific to DQN agents
class ContinualTrainer:

    def __init__(
            self, 
            env: ContinualEnv, 
            agent, 
            log_dir: str = "runs/continual_training",
            seed=1234
            ):
        self.env = env
        self.agent = agent
        self.seed = seed
        self.logger = Logger(log_dir=log_dir, args=None)  # TODO: pass args if available

        ## TODO: make a full blown logger class that saves networks etc
        # self.writer = SummaryWriter(log_dir = log_dir)

    def train(
            self, 
            update_every: int,
            eval_every: int, 
            # steps_per_epoch: int = 1000, 
            # epoch_update_freq: int = 1
        ):

        set_seed(self.seed)
        num_parallel_envs = self.env.num_parallel_envs
 
        task_rewards = dict()
        state, _ = self.env.reset()
        for i in range(
            0, 
            self.env.steps_to_do + num_parallel_envs, 
            num_parallel_envs
        ):
            with torch.no_grad():
                    action, _ = self.agent.act(state)
                
            ## TODO: create torch wrapper
            (
                next_state, 
                reward, 
                terminated, 
                truncated, 
                info
            ) = self.env.step(action.cpu().numpy())
            done = [any(i) for i in zip(terminated, truncated)]

            ## NOTE: this bit of logic is specific to dqn agents
            for j in range(num_parallel_envs):
                self.agent.record_observations(
                    state[j], 
                    action[j], 
                    reward[j], 
                    next_state[j], 
                    done[j]
                )
         
            state = next_state

            if i % update_every == 0:

                loss, value, next_value = self.agent.update_model()

                ## TODO: dqn specific
                current_epsilon = self.agent.epsilon
                self.logger.add_tensorboard(
                    f"training/task_{self.env.current_env}_loss",
                    loss.detach().cpu().numpy(),
                    i
                )
                self.logger.add_tensorboard(
                    f"training/task_{self.env.current_env}_values",
                    value.mean().detach().cpu().numpy(),
                    i
                )
                self.logger.add_tensorboard(
                    f"training/task_{self.env.current_env}_next_values",
                    next_value.mean().detach().cpu().numpy(),
                    i
                )
                self.logger.add_tensorboard(
                    f"training/task_{self.env.current_env}_epsilon",
                    current_epsilon,
                    i
                )
                self.agent.update_target_network()
            
            if i % eval_every == 0:
                tmp_task_rewards = self.evaluate(num_episodes=10)
                print(f"Evaluation at step {i}: ")
                for task, avg_reward in tmp_task_rewards.items():
                    print(f"Task {task}: Average Reward: {avg_reward}")

                    ## log rewards to tensorboard
                    self.logger.add_tensorboard(
                        f"rewards/task_{task}",
                        avg_reward, 
                        i
                    )
                print("\n====================================================\n")
                task_rewards[i] = tmp_task_rewards

            ## TODO: save networks every ...
            ## add network saving method to logger class
            if i % 1000 == 0:
                self.logger.save_network(self.agent.q_network)

        df = (
            pd.DataFrame(task_rewards)
            .T
            .reset_index()
            .rename(columns={'index':'epoch'})
            .melt(id_vars='epoch', var_name='curr_task', value_name='rewards')
        )
        return df
    
    def evaluate(self, num_episodes=10):

        results_by_task = dict()
        for task in range(self.env.num_envs):
            eval_env_spec = self.env.params[task]
            eval_env = gym.make(**eval_env_spec)
            rewards = []
            for i in range(num_episodes):
                state, _ = eval_env.reset(seed=self.seed)
                done = False
                episode_reward = 0
                while not done:
                    with torch.no_grad():
                        action, _ = self.agent.act(state, eval_mode=True)
                    (
                        next_state, 
                        reward, 
                        terminated, 
                        truncated, 
                        info
                    ) = eval_env.step(action.cpu().numpy())
                    done = terminated or truncated
                    episode_reward += reward
                    state = next_state
                # print(f"Episode {i+1}/{num_episodes}, Reward: {episode_reward}")
                rewards.append(episode_reward)
            results_by_task[task] = sum(rewards) / num_episodes

        return results_by_task
