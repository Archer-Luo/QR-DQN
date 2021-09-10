hyperparam = {
    # environment parameters
    'state_dim': 2,
    'n_actions': 2,
    'quant_num': 1,
    'start_state': [50, 100],
    'h': [3, 1],
    'rho': 0.95,
    'gamma': 0.998,

    # Learning parameters
    'nn_dimension': [20],
    'nn_activation': 'relu',
    'lr': 0.000002,

    'num_bundle': 20,

    'max_update_steps': 400000,
    'buffer_size': 1000000,
    'C': 1000,
    'epi_len': 20,
    'batch_size': 200,
    'update_freq': 20,

    'eps_initial': 0.5,
    'eps_final': 0.5,
    'eps_final_state': 0.5,
    'eps_evaluation': 0.0,
    'replay_buffer_start_size': 4000,
    'eps_annealing_states': 100000,

    'eval': True,
    'eval_min': 400,
    'eval_freq': 200,

    'eval_start': [0, 0],
    'eval_len': 10000000,
    'eval_num': 50,

    'use_per': False,
    'priority_scale': 0.7,
    'offset': 0.1,

    'soft': False,
    'soft_factor': 0.01,

    'clip': False,

    'initial_weights': None,
    'optimum': 'result095'
}
