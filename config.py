class Config:
    env_name = 'Breakout-v0'
    cnn_format = 'NCHW'
    display = False
    action_repeat = 4
    use_gpu = True
    scale = 10000
    max_step = 5000*scale
    memory_size = 100*scale
    learn_start = 5 * scale
    batch_size = 32
    discount = 0.99
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5*scale
    ep_end = 0.1
    ep_start = 1.0
    hist_len = 4
    screen_h = 84
    screen_w = 84
    train_frequency = 4
    target_q_update_step = 1 * scale
    step_input = 0
    max_delta = 1
    min_delta = -1
    