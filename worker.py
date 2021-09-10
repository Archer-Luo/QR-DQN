import ray
import tensorflow as tf
import numpy as np
from config import hyperparam
from dqn_maker import dqn_maker
from calc_epsilon import calc_epsilon
from NmodelDynamics import ProcessingNetwork
import random


def huber(x, k=10000000):
    return tf.where(tf.abs(x) < k, 0.5 * tf.pow(x, 2), k * (tf.abs(x) - 0.5 * k))


@ray.remote
class Worker:

    def __init__(self, i, replay_buffer, param_server):
        self.replay_buffer = replay_buffer
        self.param_server = param_server

        self.env = ProcessingNetwork.Nmodel_from_load(hyperparam['rho'])
        self.h = np.asarray(hyperparam['h'])

        self.dqn = dqn_maker()
        self.target_dqn = dqn_maker()

        self.replay_buffer_start_size = hyperparam['replay_buffer_start_size']
        self.max_update_steps = hyperparam['max_update_steps']
        self.C = hyperparam['C']

        # environment_info
        self.state_dim = hyperparam['state_dim']
        self.n_actions = hyperparam['n_actions']

        self.quant_num = hyperparam['quant_num']

        # buffer info
        self.batch_size = hyperparam['batch_size']
        self.use_per = hyperparam['use_per']
        self.priority_scale = hyperparam['priority_scale']

        # hyper-parameters
        self.gamma = hyperparam['gamma']

        self.current_state = np.asarray([random.randint(0, 500), random.randint(0, 500)])
        self.t = 0

        self.epi_len = hyperparam['epi_len']

        self.soft = hyperparam['soft']
        self.soft_factor = hyperparam['soft_factor']

        self.update_freq = hyperparam['update_freq']

        self.clip = hyperparam['clip']

        self.id = i

        self.actions = np.loadtxt(hyperparam['optimum'], dtype=int, delimiter=',', usecols=range(1001))

        self.losses = []

        self.tau = tf.repeat(
            tf.convert_to_tensor(np.expand_dims((2 * np.arange(self.quant_num) + 1) / (2.0 * self.quant_num), axis=0),
                                 dtype=tf.float32), axis=0, repeats=self.batch_size)

    def get_action(self, state_number, state, evaluation):
        eps = calc_epsilon(state_number, evaluation)

        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        # Otherwise, query the DQN for an action
        output = self.dqn(np.expand_dims(state, axis=0), training=False).numpy().squeeze()
        quant1 = output[0:self.quant_num]
        quant2 = output[self.quant_num:]
        expect1 = np.sum(quant1)
        expect2 = np.sum(quant2)
        action = 0 if expect1 > expect2 else 1

        return action

    def sync_dqn(self):
        new_weights = ray.get(self.param_server.get_weights.remote())
        self.dqn.set_weights(new_weights)

    def sync_target_dqn(self):
        param_weights = None
        while param_weights is None:
            param_weights = ray.get(self.param_server.sync.remote(self.id))
        self.param_server.confirm.remote(self.id)
        if not self.soft:
            self.target_dqn.set_weights(param_weights)
        else:
            new_weights = [(1 - self.soft_factor) * x + self.soft_factor * y for x, y in
                           zip(self.target_dqn.get_weights(), param_weights)]
            self.target_dqn.set_weights(new_weights)

    def run(self):
        # time.sleep(random.randint(0, 10))
        self.dqn.set_weights(ray.get(self.param_server.get_weights.remote()))
        self.target_dqn.set_weights(self.dqn.get_weights())

        while self.t < self.replay_buffer_start_size + 1:
            action = self.get_action(self.t, self.current_state, False)
            next_state = self.env.next_state_N1(self.current_state, action)
            reward = -(self.current_state @ self.h)
            self.replay_buffer.add_experience.remote(action, self.current_state, next_state, reward)  # TODO

            if self.t % self.epi_len == 0:
                self.current_state = np.asarray([random.randint(0, 500), random.randint(0, 500)])
            else:
                self.current_state = next_state

            # self.sync_dqn()

            self.t += 1

        while self.t < self.max_update_steps:
            action = self.get_action(self.t, self.current_state, False)
            next_state = self.env.next_state_N1(self.current_state, action)
            reward = -(self.current_state @ self.h)
            self.replay_buffer.add_experience.remote(action, self.current_state, next_state, reward)  # TODO

            if self.t % self.epi_len == 0:
                self.current_state = np.asarray([random.randint(0, 500), random.randint(0, 500)])
            else:
                self.current_state = next_state

            self.sync_dqn()

            if self.t % self.update_freq == 0:
                if self.use_per:
                    (states, actions, rewards, new_states), importance, indices = ray.get(
                        self.replay_buffer.get_minibatch.remote(
                            batch_size=self.batch_size, priority_scale=self.priority_scale))
                    importance = tf.cast(importance ** (1 - calc_epsilon(self.t, False)), dtype=tf.float32)
                else:
                    states, actions, rewards, new_states = ray.get(self.replay_buffer.get_minibatch.remote(
                        batch_size=self.batch_size, priority_scale=self.priority_scale))

                # Target DQN estimates q-vals for new states
                target_values = self.target_dqn(new_states, training=False).numpy()
                target_values = target_values.reshape((self.batch_size, self.n_actions, self.quant_num))
                target_future_actions = np.argmax(np.sum(target_values, axis=2).squeeze(), axis=1)
                target_future_v = target_values[range(self.batch_size), target_future_actions].squeeze()

                new_states_clip = np.minimum(new_states, np.full((self.batch_size, self.state_dim), 999))
                optimum_actions = self.actions[new_states_clip[:, 0], new_states_clip[:, 1]] - 1

                correct = np.sum(target_future_actions == optimum_actions)
                self.param_server.add_sample.remote(len(optimum_actions), correct)

                # Calculate targets (bellman equation)
                target_q = np.repeat(np.expand_dims(rewards, axis=1), self.quant_num, axis=1) + (
                        self.gamma * target_future_v)

                # Use targets to calculate loss (and use loss to calculate gradients)
                with tf.GradientTape() as tape:
                    preds = self.dqn(states)
                    q_values = tf.reshape(preds, [self.batch_size, self.n_actions, self.quant_num])
                    Q = tf.gather_nd(params=q_values, indices=tf.stack((tf.constant(range(self.batch_size), tf.int32),
                                                                        tf.constant(actions, tf.int32)), -1))
                    error = target_q - Q
                    loss = huber(error) * tf.abs(self.tau - tf.cast(error < 0, tf.float32))

                    if self.use_per:
                        # Multiply the loss by importance, so that the gradient is also scaled.
                        # The importance scale reduces bias against situataions that are sampled
                        # more frequently.
                        loss = tf.reduce_mean(
                            loss * tf.repeat(tf.expand_dims(importance, axis=1), repeats=self.quant_num, axis=1))
                    else:
                        loss = tf.reduce_mean(loss)

                model_gradients = tape.gradient(loss, self.dqn.trainable_variables)

                if self.clip:
                    model_gradients = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in model_gradients]

                self.param_server.update_weights.remote(model_gradients)

                if self.use_per:
                    self.replay_buffer.set_priorities.remote(indices, tf.reduce_mean(tf.abs(error), axis=1))

                # print(('{}' + ': ' + '{:10.5f}').format(self.t, loss), flush=True)

            self.t += 1

            if self.t % self.C == 0:
                self.sync_target_dqn()

            if self.t % self.C == 1:
                self.losses.append(loss)
                print(('{}' + ': ' + '{:10.5f}').format(self.t, loss), flush=True)

        record, outside = ray.get(self.replay_buffer.get_record.remote())
        percentages = ray.get(self.param_server.get_percentages.remote())
        final_weights = ray.get(self.param_server.get_weights.remote())

        return final_weights, record, outside, percentages, self.losses
