import gymnasium as gym

class ContinualEnv(gym.Env):

    def __init__(self, params: list[dict], steps_per_env: int, num_parallel_envs: int = 1):
        self.params = params
        self.num_parallel_envs = num_parallel_envs
        self.steps_per_env = steps_per_env
        assert self.steps_per_env%self.num_parallel_envs == 0, "steps_per_env must be divisible by num_envs"

        self.envs = [
            gym.make_vec(num_envs = self.num_parallel_envs, **param) for param in self.params
        ]
        self.num_envs = len(self.envs)
        self.current_env = 0
        self.current_step = 0
        self.total_steps = 0
        self.steps_to_do = self.steps_per_env * self.num_envs
        self.current_env_instance = self.envs[self.current_env]
        self.action_space = self.current_env_instance.action_space
        self.observation_space = self.current_env_instance.observation_space
    
    def reset(self, seed=None):
        return self.current_env_instance.reset()
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.current_env_instance.step(action)
        self.current_step += 1*self.num_parallel_envs
        self.total_steps += 1*self.num_parallel_envs
        
        if self.current_step >= self.steps_per_env:
            print(f"Switching to next environment: {self.current_env} -> {self.current_env + 1} at step {self.current_step}")
            self.current_env += 1
            if self.current_env < len(self.envs):
                self.current_env_instance = self.envs[self.current_env]
                self.current_step = 0
                ## reset the environment so that the next step starts fresh
                next_state, _ = self.current_env_instance.reset()
            else:
                print("No more environments to switch to. Continuing with the last environment.")
                ## TODO: handle end of environments gracefully
        
        return next_state, reward, terminated, truncated, info