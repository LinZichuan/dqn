import gym
import cv2

class GymEnv:
    def __init__(self, config):
        self.env = gym.make(config.env_name)
        self._screen = None
        self.reward = 0
        self.term = True
        self.display = config.display
        self.dims = (config.screen_h, config.screen_w)

    def newGame(self):
        self._screen = self.env.reset()
        self._screen, self.reward, self.term, _ = self.env.step(0)
        if self.display:
            self.render()
        return self.screen, 0, 0, self.term

    def step(self, action):
        self._screen, self.reward, self.term, _ = self.env.step(action)
        return self.screen, self.reward, self.term

    def render(self):
        self.env.render()

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def screen(self):
        return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_RGB2GRAY)/255.0, self.dims)

