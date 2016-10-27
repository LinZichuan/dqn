import gym

class GymEnv:
    def __init__(self, config):
        self.env = gym.make(config.env_name)
        self.screen = None
        self.reward = 0
        self.term = True
        self.display = config.display

    def newGame(self):
        self.screen = self.env.reset()
        self.screen, self.reward, self.term, _ = self.env.step(0)
        if self.display:
            self.render()
        return self.screen, 0, 0, self.term

    def step(self, action):
        self.screen, self.reward, self.term, _ = self.env_step(action)

    @property
    def action_size(self):
        return self.env.action_space.n
