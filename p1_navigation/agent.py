class DQNAgent:
    def __init__(self, action_space):
        self._action_space = action_space

    def act(self, observation):
        return self._action_space.sample()

    def step(self, obs, action, reward, next_obs, done):
        pass

    def train(self):
        pass
