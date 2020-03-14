import gym

from udacity_rl.epsilon import EpsilonExpDecay


def assert_that(obj, matcher):
    assert matcher(obj)


class ContractMatcher:
    def __init__(self, interface, properties):
        self._interface = interface
        self._properties = properties
        self._tested_obj = None
        self._missing_functions = []
        self._missing_properties = []

    def __call__(self, obj):
        self._tested_obj = obj
        self._collect_missing_functions()
        self._collect_missing_properties()
        return len(self._missing_functions) == 0 and len(self._missing_properties) == 0

    def _collect_missing_properties(self):
        for p in self._properties:
            if getattr(self._tested_obj, p) is None:
                self._missing_properties.append(p)

    def _collect_missing_functions(self):
        for func, args in self._interface:
            try:
                getattr(self._tested_obj, func)(*args)
            except (AttributeError, NotImplementedError):
                self._missing_functions.append(func)

    def __repr__(self):
        return f"the object {self._tested_obj}, is missing{self._print_missing()}"

    def _print_missing(self):
        message = ""
        if len(self._missing_functions):
            message += f" these functions: {', '.join(self._missing_functions)}"
        if len(self._missing_properties):
            if len(message) != 0:
                message += " and"
            message += f" these properties: {', '.join(self._missing_properties)}"
        return message


def follows_contract(interface=None, properties=None):
    return ContractMatcher(interface or [], properties or [])


class GymSession(gym.Wrapper):
    def __init__(self, gym_instance, eps_calc):
        super().__init__(gym_instance)
        self.eps_calc = eps_calc

    def test(self, agent, num_episodes=100):
        return self._run_session(agent, num_episodes)

    def train(self, agent, num_episodes=1000, train_freq=4):
        return self._run_session(agent, num_episodes, train_freq)

    def _run_session(self, agent, num_episodes, train_freq=None):
        average_reward = 0
        for _ in range(num_episodes):
            done = False
            obs = self.env.reset()
            total_r = 0
            t = 0
            while not done:
                a = agent.act(obs, self.eps_calc.epsilon)
                next_obs, r, done, _ = self.env.step(a)
                agent.step(obs, a, r, next_obs, done)
                if train_freq and t % train_freq == 0:
                    agent.train()
                obs = next_obs
                total_r += r
                self.eps_calc.update()
                t += 1
            average_reward += total_r

        return average_reward / num_episodes


def train_to_target(agent, random_walk, target_score):
    max_episodes = 10
    episode = 0
    score = random_walk.reward_range[0]
    while score < target_score and episode < max_episodes:
        random_walk.train(agent, train_freq=10)
        score = random_walk.test(agent)
        episode += 1
    assert episode < max_episodes
