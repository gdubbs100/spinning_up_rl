import torch
import pandas as pd

from environments.continual_environment import ContinualEnv
from utils.training_utils import set_seed

class ContinualTrainer:

    def __init__(
            self, 
            env: ContinualEnv, 
            agent, 
            logger=None,
            seed=1234
            ):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.seed = seed

    def train(
            self, 
            steps_per_update: int, 
            # steps_per_epoch: int = 1000, 
            # epoch_update_freq: int = 1
        ):

        set_seed(self.seed)
        steps_per_env = self.env.steps_per_env
        num_parallel_envs = self.env.num_parallel_envs
 
        rewards = dict()
        losses = dict()
        curr_task = dict()
        epoch_reward = 0
        state, _ = self.env.reset()
        for i in range(self.env.steps_to_do // self.env.num_parallel_envs):
            with torch.no_grad():
                    try:
                        action, _ = self.agent.act(state)
                    except:
                        breakpoint()
                
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

            epoch_reward += reward.sum()
            # self.logger.log(state, action, reward, next_state, done, info)
            
            state = next_state

            ## TODO: this is DQN specific
            self.agent.update_target_network()

            if i % steps_per_update == 0:
                # breakpoint()
                loss, _, _ = self.agent.update_model()
                rewards[i] = epoch_reward / steps_per_update
                losses[i] = loss.detach().cpu().numpy()
                curr_task[i] = self.env.current_env
                # print(f"Step {i+1}, Environment {self.env.current_env}:")
                # print(f"Step loss: {loss}, step reward: {epoch_reward / steps_per_update}")
                epoch_reward = 0 # reset epoch reward

        df = pd.DataFrame({
            'epoch': range(len(rewards)),
            'rewards': pd.Series(rewards),
            'losses': pd.Series(losses),
            'curr_task': pd.Series(curr_task)
        })
        return df

        # for epoch in range(num_epochs):
        #     state, _ = self.env.reset()
        #     epoch_reward = 0
        #     for _ in range(steps_per_epoch):
        #         with torch.no_grad():
        #             try:
        #                 action, _ = self.agent.act(state)
        #             except:
        #                 breakpoint()
                
        #         ## TODO: create torch wrapper
        #         next_state, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
        #         done = [any(i) for i in zip(terminated, truncated)]

        #         ## NOTE: this bit of logic is specific to dqn agents
        #         for i in range(num_parallel_envs):
        #             self.agent.record_observations(
        #                 state[i], 
        #                 action[i], 
        #                 reward[i], 
        #                 next_state[i], 
        #                 done[i]
        #             )

        #         epoch_reward += reward.sum()
        #         # self.logger.log(state, action, reward, next_state, done, info)
                
        #         state = next_state
            
        #     loss, _, _ = self.agent.update_model()
        #     rewards[epoch]=epoch_reward / steps_per_epoch
        #     losses[epoch]=loss.detach().cpu().numpy()
        #     curr_task[epoch] = self.env.current_env
        #     print(f"Epoch {epoch+1}, Environment {self.env.current_env}:")
        #     print(f"Epoch loss: {loss}, epoch reward: {epoch_reward / steps_per_epoch}")
        #     ## TODO: log agent performance, maybe save model
        #     ## TODO: evaluate agent performance on all environments
        # df = pd.DataFrame({
        #     'rewards': pd.Series(rewards),
        #     'losses': pd.Series(losses),
        #     'curr_task': pd.Series(curr_task)
        # }, index=range(num_epochs))
        # return df