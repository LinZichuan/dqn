import tensorflow as tf
from config import Config
from gymenv import GymEnv
from agent import Agent

def main():
    with tf.Session() as sess:
        config = Config()
        
        env = GymEnv(config)
        agent = Agent(config, env, sess)
        agent.train()
        #agent.play(n_episode=1, n_step=10000)


if __name__ == '__main__':
    main()
