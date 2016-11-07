import numpy as np
import random
import os
from ops import conv2d, linear
from replay_memory import ReplayMemory
import tensorflow as tf
from tqdm import tqdm
import time

class Agent:
    def __init__(self, config, env, sess):
        self.sess = sess
        self.env = env
        self.env_name = config.env_name
        self.env_type = config.env_type
        self.cnn_format = config.cnn_format
        self.batch_size, self.hist_len,  self.screen_h, self.screen_w = \
                config.batch_size, config.hist_len, config.screen_h, config.screen_w
        self.train_frequency = config.train_frequency
        self.target_q_update_step = config.target_q_update_step
        self.max_step = config.max_step
        self.test_step = config.test_step
        self.learn_start = config.learn_start
        self.min_delta = config.min_delta
        self.max_delta = config.max_delta
        self.learning_rate_minimum = config.learning_rate_minimum
        self.learning_rate = config.learning_rate
        self.learning_rate_decay_step = config.learning_rate_decay_step
        self.learning_rate_decay = config.learning_rate_decay
        self.is_train = config.is_train
        self.display = config.display
        self.double_q = config.double_q
        self.dueling = config.dueling

        if self.is_train:
            self.memory = ReplayMemory(config)
        self.history = np.zeros([self.hist_len, self.screen_h, self.screen_w], dtype=np.float32)

        self.ep_end = config.ep_end
        self.ep_start = config.ep_start
        self.ep_end_t = config.ep_end_t
        self.min_reward = config.min_reward
        self.max_reward = config.max_reward
        self.discount = config.discount

        self.step_op = tf.Variable(0, trainable=False, name='step')
        self.checkpoint_dir = os.path.join('checkpoints/', config.model_dir)
        self.summary_log_path = os.path.join('logs/', config.model_dir)

        self.build_graph()

    def train(self):
        start_step = self.step_op.eval()

        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []
        screen, reward, action, term = self.env.newRandomGame()

        for i in xrange(self.hist_len):
            self.history[i] = screen

        for self.step in tqdm(xrange(start_step, self.max_step), ncols=70, initial=start_step):
            if self.step == self.learn_start:
                num_game, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                max_avg_ep_reward = 0
                ep_rewards, actions = [], []
                #new game? because we start learning from middle of a game episode.

            action = self.predict(self.history)
            screen, reward, term = self.env.act(action)
            self.observe(screen, reward, action, term)

            if term:
                screen, reward, action, term = self.env.newRandomGame()
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
                    print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                            % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)
                    if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                        self.step_op.assign(self.step + 1).eval()
                        self.save_model(self.step + 1)
                        self.memory.save()
                        #self.step_assign_op.eval({self.step_input: self.step + 1})
                        max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)
                    if self.step > 180:
                        #self.learning_rate_op.eval({self.learning_rate_step: self.step})
                        #inject summary
                        self.inject_summary({
                            'avg.reward': avg_reward,
                            'avg.loss': avg_loss,
                            'avg.q': avg_q,
                            'episode.max_reward': max_ep_reward,
                            'episode.min_reward': min_ep_reward,
                            'episode.avg_reward': avg_ep_reward,
                            'episode.num_of_game': num_game,
                            'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
                        })
                    num_game, self.update_count, ep_reward = 0, 0, 0.
                    total_reward, self.total_loss, self.total_q = 0., 0., 0.
                    ep_rewards, actions = [], []

    def predict(self, s_t, test_ep=None):
        ep = test_ep or (self.ep_end + max(0., (self.ep_start - self.ep_end) * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
        if random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            action = self.q_action.eval({self.s_t: [s_t]})[0]
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
        gym_dir = './video/%s-%s' % (self.env_name, time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime()))
        self.env.env.monitor.start(gym_dir)
        test_history = np.zeros([self.hist_len, self.screen_h, self.screen_w], dtype=np.float32)
        best_reward = 0
        for idx in xrange(n_episode):
            self.env.env.reset()
            screen, reward, action, term = self.env.newRandomGame()
            current_reward = 0
            for i in xrange(self.hist_len):
                test_history[i] = screen
            for s in xrange(n_step):
                #action = self.env.action_space_sample()
                action = self.predict(test_history, test_ep=0.05)
                screen, reward, term = self.env.act(action, is_training=False)
                test_history[:-1] = test_history[1:]
                test_history[-1] = screen
                current_reward += reward
                if self.display:
                    self.env.render()
                if term:
                    print 'step: %d' % s
                    break
            best_reward = max(best_reward, current_reward)
            print 'current_reward: %d, best_reward: %d' % (current_reward, best_reward)
        self.env.env.monitor.close()

    def createQNetwork(self, scope_name):
        init = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu
        w = {}

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

            if self.dueling:
                value_hid, w['l4_w'], w['l4_b'] = linear(l3_flat, 512, activation_fn=activation_fn, name='value_hid')
                adv_hid, w['l4_adv_w'], w['l4_adv_b'] = linear(l3_flat, 512, activation_fn=activation_fn, name='adv_hid')
                value, w['val_w_out'], w['val_b_out'] = linear(value_hid, 1, name='value_out')
                advantage, w['adv_w_out'], w['adv_b_out'] = linear(adv_hid, self.env.action_size, name='adv_out')
                # Average Dueling
                q = value + (advantage - tf.reduce_mean(advantage, reduction_indices=1, keep_dims=True))
            else:
                l4, w['l4_w'], w['l4_b'] = linear(l3_flat, 512, activation_fn=activation_fn, name='l4')
                q, w['q_w'], w['q_b'] = linear(l4, self.env.action_size, name='q')

            return s_t, w, q

    def build_graph(self):
        ###
        self.s_t, self.w, self.q = self.createQNetwork('prediction') ##self.q = max Q value
        self.q_action = tf.argmax(self.q, dimension=1)
        self.target_s_t, self.t_w, self.target_q = self.createQNetwork('target')
        self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
        self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        q_summary = []
        avg_q = tf.reduce_mean(self.q, 0)
        for idx in xrange(self.env.action_size):
            q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
        self.q_summary = tf.merge_summary(q_summary, 'q_summary')

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted
            self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')
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
            #self.optim = tf.train.RMSPropOptimizer(0.00025).minimize(self.loss)
            #self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['avg.reward', 'avg.loss', 'avg.q', \
                    'episode.max_reward', 'episode.min_reward', 'episode.avg_reward', \
                    'episode.num_of_game', 'training.learning_rate']
            self.summary_placeholders = {}
            self.summary_ops = {}
            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.scalar_summary("%s-%s/%s" % \
                        (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

            hist_summary_tags = ['episode.rewards', 'episode.actions']
            for tag in hist_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.histogram_summary(tag, self.summary_placeholders[tag])


            self.writer = tf.train.SummaryWriter(self.summary_log_path, self.sess.graph)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)
        self.load_model()
        if self.is_train:
            self.memory.load()
        self.update_target_q_network()

    def inject_summary(self, tag_dict):
        print 'inject summary!'
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, self.step)

    def load_model(self):
        print ("[*] Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print ("[*] Load SUCCESS: %s" % fname)
            return True
        else:
            print ("[*] Load FAILED: %s" % self.checkpoint_dir)
            return False

    def save_model(self, step):
        print ("[*] Saving checkpoints...")
        model_name = type(self).__name__
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

    def q_learning_mini_batch(self):
        if self.memory.count < self.hist_len:
            return
        else:
            s_t, action, reward, s_t_plus_1, term = self.memory.sample()

        if self.double_q:
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})
            term = np.array(term) + 0.0
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
                self.target_s_t: s_t_plus_1,
                self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
            })
            target_q_t = (1 - term) * self.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
            term = np.array(term) + 0.0
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = (1 - term) * self.discount * max_q_t_plus_1 + reward

        _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
            self.s_t: s_t,
            self.target_q_t: target_q_t,
            self.action: action,
            self.learning_rate_step: self.step,
        })

        self.writer.add_summary(summary_str, self.step)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def update_target_q_network(self):
        print "update target network!"
        for name in self.w.keys():
            self.t_w[name].assign(self.w[name]).eval()
            #self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    '''def create_copy_op(self):
        copy_ops = []
        for name in self.w.keys():
            copy_op = self.t_w[name].assign(self.w[name])
            copy_ops.append(copy_op)
        self.copy_op = copy_ops'''

