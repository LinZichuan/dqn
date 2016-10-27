import numpy as np
import random
from ops import conv2d, linear
from replay_memory import ReplayMemory
import tensorflow as tf

class Agent:
    def __init__(self, config, env, sess):
        self.sess = sess
        self.env = env
        self.cnn_format = config.cnn_format
        self.batch_size, self.hist_len,  self.screen_h, self.screen_w = \
                config.batch_size, config.hist_len, config.screen_h, config.screen_w
        self.train_frequency = config.train_frequency
        self.target_q_update_step = config.target_q_update_step
        self.step_input = config.step_input
        self.max_step = config.max_step
        self.learn_start = config.learn_start
        self.min_delta = config.min_delta
        self.max_delta = config.max_delta
        self.learning_rate_minimum = config.learning_rate_minimum
        self.learning_rate = config.learning_rate,
        self.learning_rate_decay_step = config.learning_rate_decay_step
        self.learning_rate_decay = config.learning_rate_decay

        self.memory = ReplayMemory(config)
        self.history = np.zeros([self.hist_len, self.screen_h, self.screen_w], dtype=np.float32)

        self.build_graph()

    def train(self):
        start_step = self.step_input

        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []
        screen, reward, action, term = self.env.newGame()

        for i in xrange(self.hist_len):
            self.history[i] = screen

        for self.step in xrange(start_step, self.max_step):
            if self.step == self.learn_start:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                max_avg_ep_reward = 0
                ep_rewards, actions = [], []
                #new game? because we start learning from middle of a game episode.

            action = self.predict(self.history)
            screen, reward, term, _ = self.env.step(action)
            self.observer(screen, reward, action, term)

            if term:
                screen, reward, action, term = self.env.newGame()
                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward

            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1:
                    avg_reward = total_reward / self.test_step
                    avg_loss = self.total_loss / self.update_count  ##total_loss is updated in q_learn_mini_batch
                    avg_q = self.total_q / self.update_count  ##q is updated in q_learn_mini_batch
                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0
                    print '\navg_r: %.4f, avg_l: %.6f, avg_q: %.3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                            % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward)
                    if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                        self.step_assign_op.eval({self.step_input: self.step + 1})
                        self.save_model(self.step + 1)
                        max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)
                    if self.step > 180:
                        #inject summary
                        pass
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                max_avg_ep_reward = 0
                ep_rewards, actions = [], []

    def predict(self, s_t, test_ep=None):
        ep = test_ep or (self.ep_end + max(0., self.ep_start - self.ep_end) * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t)
        if random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            action = self.q_action.eval({self.s_t: s_t})[0]
        return action

    def observe(self, screen, reward, action, term):
        #add to history, memory
        #q_learn, update_target_q
        reward = max(self.min_reward, min(self.max_reward, reward))
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen
        self.memory.add(screen, reward, action, term)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()
            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()


    def play(self, n_step=10000, n_episode=1):
        test_history = np.zeros([self.hist_len, self.screen_h, self.screen_w], dtype=np.float32)
        for idx in xrange(n_episode):
            screen, reward, action, term = self.env.newGame()
            current_reward, best_reward = 0, 0
            for i in xrange(self.hist_len):
                test_history[i] = screen
            for s in xrange(n_step):
                #action = self.env.action_space_sample()
                action = predict(test_history, test_ep=0.05)
                screen, reward, term, _ = self.env.step(action)
                current_reward += reward
                if self.display:
                    self.env.render()
                if term:
                    break
            best_reward = max(best_reward, current_reward)
            print 'current_reward: %d, best_reward: %d' % (current_reward, best_reward)

    def createQNetwork(self, s_t, w, q, scope_name):
        init = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        with tf.variable_scope(scope_name):
            if self.cnn_format == 'NHWC':
                s_t = tf.placeholder('float32', 
                        [None, self.screen_h, self.screen_w, self.hist_len], name='s_t')
            else:
                s_t = tf.placeholder('float32',
                        [None, self.hist_len, self.screen_h, self.screen_w], name='s_t')
            l1, w['l1_w'], w['l1_b'] = conv2d(s_t,
                    32, [8,8], [4,4], init, activation_fn, self.cnn_format, name='l1')
            l2, w['l2_w'], w['l2_b'] = conv2d(l1,
                    64, [4,4], [2,2], init, activation_fn, self.cnn_format, name='l2')
            l3, w['l3_w'], w['l3_b'] = conv2d(l2,
                    64, [3,3], [1,1], init, activation_fn, self.cnn_format, name='l3')

            shape = l3.get_shape().as_list()
            l3_flat = tf.reshape(l3, [-1, reduce(lambda x,y: x*y, shape[1:])])

            l4, w['l4_w'], w['l4_b'] = linear(l3_flat, 512, activation_fn=activation_fn, name='l4')
            q, w['q_w'], w['q_b'] = linear(l4, self.env.action_size, name='q')
        

    def build_graph(self):
        self.w = {}
        self.t_w = {}
        self.q, self.target_q = np.empty([self.env.action_size], dtype=np.float32), np.empty([self.env.action_size], dtype=np.float32)
        self.s_t, self.target_s_t = None, None

        ###
        self.createQNetwork(self.s_t, self.w, self.q, 'prediction') ##self.q = max Q value
        self.q_action = tf.argmax(self.q, dimension=1)
        self.createQNetwork(self.target_s_t, self.t_w, self.target_q, 'target')

        avg_q = tf.reduce_mean(self.q, 0)
        q_summary = []
        for idx in xrange(self.env.action_size):
            q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
        self.q_summary = tf.merge_summary(q_summary, 'q_summary')

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')


            delta = self.target_q_t - q_acted
            self.clipped_delta = tf.clip_by_value(delta, self.min_delta, self.max_delta, name='clipped_delta')
            self.global_step = tf.Variable(0, trainable=False)

            self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                    tf.train.exponential_decay(
                        self.learning_rate,
                        self.learning_rate_step,
                        self.learning_rate_decay_step,
                        self.learning_rate_decay,
                        staircase=True))
            self.optim = tf.train.RMSPropOptimizer(
                    self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)
        tf.initialize_all_variables().run()
        self.update_target_q_network()

    def q_learning_mini_batch(self):
        if self.memory.count < self.hist_len:
            return
        else:
            s_t, action, reward, s_t_plus_1, term = self.memory.sample()

        q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
        term = np.array(term) + 0.0
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        target_q_t = (1 - term) * self.discount * max_q_t_plus_1 + reward

        _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t,
            self.learning_rate_step: self.step
        })
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w[name].assign(self.w[name].eval())
            #self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})
