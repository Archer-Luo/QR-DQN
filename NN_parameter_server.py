from dqn_maker import dqn_maker
import ray
from config import hyperparam
import numpy as np


@ray.remote
class NNParamServer:
    def __init__(self):
        self.model = dqn_maker()
        self.sync_request = np.zeros(hyperparam['num_bundle'])
        self.param_weights = None
        self.processing = False
        self.count = 0
        self.correct = 0
        self.percentages = []
        self.update_num = 0

        self.eval = hyperparam['eval']
        self.eval_min = hyperparam['eval_min']
        self.eval_freq = hyperparam['eval_freq']
        self.eval_num = hyperparam['eval_num']

    def get_weights(self):
        return self.model.get_weights()

    def update_weights(self, gradient):
        self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

    def update_weights_list(self, gradient_list):
        for gradient in gradient_list:
            self.update_weights(gradient)

    def sync(self, i):
        self.sync_request[i] = 1
        if np.isin(self.sync_request, [1, 2]).all():
            if self.param_weights is None:
                self.update_num += 1
                self.param_weights = self.model.get_weights()

                return self.param_weights
            else:
                return self.param_weights
        else:
            return None

    def confirm(self, i):
        self.sync_request[i] = 2
        if np.isin(self.sync_request, [2]).all():
            self.sync_request = np.zeros(hyperparam['num_bundle'])
            if self.eval and self.update_num >= self.eval_min and self.update_num % self.eval_freq == 0:
                # self.evaluation(self.param_weights)
                index = int((self.update_num - self.eval_min) / self.eval_freq)
                self.model.save_weights(f'checkpoints/checkpoint_{index}')
            self.param_weights = None
            if self.count != 0:
                percentage = self.correct / self.count * 100
                self.percentages.append(percentage)
                print(str(self.update_num) + ': ' + str(percentage))
            self.count = 0
            self.correct = 0

    def add_sample(self, num_sample, num_correct):
        self.count += num_sample
        self.correct += num_correct

    def get_percentages(self):
        return self.percentages
