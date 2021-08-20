from config import hyperparam
from replay_buffer import ReplayBuffer
import ray
from worker import Worker
import time
import numpy as np
from dqn_maker import dqn_maker
from NN_parameter_server import NNParamServer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import errno


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def main():
    parameter_server = NNParamServer.remote()
    replay_buffer = ReplayBuffer.remote()
    workers = [Worker.remote(i, replay_buffer, parameter_server) for i in range(hyperparam['num_bundle'])]
    ready_id, remaining_ids = ray.wait([worker.run.remote() for worker in workers],
                                       num_returns=hyperparam['num_bundle'])
    final_weights, record, outside, percentages, losses = ray.get(ready_id[hyperparam['num_bundle'] - 1])

    fig1 = plt.figure(1)
    plt.plot(losses)
    plt.ylabel('loss')

    fig2 = plt.figure(2)
    plt.plot(percentages)
    plt.ylabel('percentage accuracy')

    filename = 'figures/fig_2.pdf'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    multipage(filename, [fig1, fig2], dpi=250)

    plt.show()

    with open('record', 'w') as f:
        np.savetxt(f, record, fmt='%i', delimiter=",")

    print('outside: {0}'.format(outside), flush=True)

    evaluate_dqn = dqn_maker()
    evaluate_dqn.set_weights(final_weights)
    action_result = np.empty([151, 151])
    # v_result = np.empty([151, 151])
    # difference = np.empty([151, 151])

    evaluate_dqn.save_weights('final_weights_2')
    quant_num = hyperparam['quant_num']

    for a in range(151):
        for b in range(151):
            state = np.array([a, b])
            output = evaluate_dqn(np.expand_dims(state, axis=0), training=False).numpy().squeeze()
            quant1 = output[0:(quant_num - 1)]
            quant2 = output[quant_num:]
            expect1 = np.sum((1.0 / quant_num) * quant1)
            expect2 = np.sum((1.0 / quant_num) * quant2)
            action = 0 if expect1 < expect2 else 1
            action_result[a, b] = action

    with open('rho{0}_gamma{1}_action'.format(hyperparam['rho'], hyperparam['gamma']), 'w') as f:
        np.savetxt(f, action_result, fmt='%i', delimiter=",")

    # with open('difference', 'w') as f:
    #     np.savetxt(f, difference, fmt='%10.5f', delimiter=",")
    #
    # with open('rho{0}_gamma{1}_value'.format(hyperparam['rho'], hyperparam['gamma']), 'w') as f:
    #     np.savetxt(f, v_result, fmt='%10.5f', delimiter=",")


start_time = time.time()

ray.init()

if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))
