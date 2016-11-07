import tensorflow as tf
from config import Config
from gymenv import GymEnv
from agent import Agent

def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        config = Config()
        
        env = GymEnv(config)
        agent = Agent(config, env, sess)
        if config.is_train:
            agent.train()
        else:
            agent.play(n_episode=100, n_step=10000)


if __name__ == '__main__':
    main()
