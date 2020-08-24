

import numpy as np
import matplotlib.pyplot as plt
'''
states = ('Healthy', 'Fever')
end_state = 'E'
observations = ['normal', 'cold', 'dizzy']
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {
    'Healthy': {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
    'Fever': {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01}}
emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}}

'''
M = 2
N = 3
states = ('first', 'second', 'third')

observations = ('two', 'one', 'two', 'one', 'one', 'one', 'one', 'one', 'one', 'two')
A_mod1 = {'first': {'first': 0.9, 'second': 0.05, 'third': 0.05},
          'second': {'first': 0.45, 'second': 0.1, 'third': 0.45},
          'third': {'first': 0.45, 'second': 0.45, 'third': 0.1}}

A_mod2 = {'first': {'first': 0.5, 'second': 0.25, 'third': 0.25},
          'second': {'first': 0.2, 'second': 0.4, 'third': 0.4},
          'third': {'first': 0.1, 'second': 0.45, 'third': 0.45}}

B = {'first': {'one': 0.5, 'two': 0.5},
     'second': {'one': 0.75, 'two': 0.25},
     'third': {'one': 0.25, 'two': 0.75}}

pi = {'first': 0.333, 'second': 0.333, 'third': 0.333}

# states = np.array(('Healthy', 'Fever'), 'str')
# observations = np.array(('normal', 'cold', 'dizzy'), 'str')
# start_probs = np.array((0.6, 0.4), 'float')
# trans_probs = np.array(((0.69, 0.3, 0.01),(0.4, 0.59, 0.01)), 'float')
# emis_probs = np.array(((0.5, 0.4, 0.1),(0.1, 0.3, 0.6)), 'float')


# forward_backward_procedure
def forward_backward_procedure(obs, states, start_prob, transition_prob, emis_prob):
    # FARWARD STEP
    # initialize alpha, c0
    alpha = {state: np.zeros(len(obs)) for state in states}
    scale_factor = np.zeros(len(obs))
    # compute alpha_0(i)
    for state in states:
        alpha[state][0] = start_prob[state] * emis_prob[state][obs[0]]
        scale_factor[0] += alpha[state][0]
    # scaling alpha_0(i)
    scale_factor[0] = 1 / scale_factor[0]
    for state in states:
        alpha[state][0] = scale_factor[0] * alpha[state][0]
    # compute alpha_t(i)
    for t in range(1, len(obs)):
        for i in states:
            for j in states:
                alpha[i][t] += alpha[j][t - 1] * transition_prob[j][i]
            alpha[i][t] *= emis_prob[i][obs[t]]
            scale_factor[t] += alpha[i][t]
        # scale alpha_t(i)
        scale_factor[t] = 1 / scale_factor[t]
        for i in states:
            alpha[i][t] *= scale_factor[t]

    # BACKWARD STEP
    # initialize beta_T scaled
    T = len(obs) - 1
    beta = {state: np.zeros(len(obs)) for state in states}
    for state in states:
        beta[state][T] = scale_factor[T]
    # compute beta_t(i)
    for t in range(len(obs) - 2, -1, -1):
        for i in states:
            for j in states:
                beta[i][t] += transition_prob[i][j] * emis_prob[j][observations[t + 1]] * beta[j][t + 1]
            # scale beta_t(i)
            beta[i][t] *= scale_factor[t]

    return alpha, beta, scale_factor


alpha, beta, scale_factor = forward_backward_procedure(observations, states, pi, A_mod1, B)
# print('alpha', alpha, 'beta', beta, scale_factor)

def gamma_and_gamma_double(obs, states, alpha, beta, trans_prob, emis_prob):
    # initialize gamma_t(i) and gamma_t(i, j)
    gamma_double = {state: {s: np.zeros(len(obs)) for s in states} for state in states}
    gamma = {state: np.zeros(len(obs)) for state in states}
    # initialize gamma_T(i)
    T = len(obs) - 1
    for state in states:
        gamma[state][T] = alpha[state][T]
    # compute gamma_t(i) and gamma_t(i, j)
    for t in range(len(obs) - 1):
        for i in states:
            for j in states:
                gamma_double[i][j][t] = alpha[i][t] * trans_prob[i][j] * emis_prob[j][obs[t + 1]] * beta[j][
                    t + 1]
                gamma[i][t] += gamma_double[i][j][t]
    return gamma, gamma_double


g, g_d = gamma_and_gamma_double(observations, states, alpha, beta, A_mod1, B)
# print('gamma', g, 'gamma_double', g_d)

def baum_welch_algorithm(obs, states, alpha, beta, trans_prob, emis_prob):
    # COMPUTE GAMMA
    gamma, gamma_double = gamma_and_gamma_double(observations, states, alpha, beta, trans_prob, B)

    # RE-ESTIMATE START_PROB
    new_start_prob = {state: 0 for state in states}
    for i in states:
        new_start_prob[i] = gamma[i][0]

    # RE-ESTIMATE TRANS_PROB
    new_trans_prob = {state: {s: 0 for s in states} for state in states}
    for i in states:
        denom = 0
        for t in range(len(obs) - 1):
            denom += gamma[i][t]
        for j in states:
            numer = 0
            for t in range(len(obs) - 1):
                numer += gamma_double[i][j][t]
            new_trans_prob[i][j] = numer / denom

    # RE-ESTIMATE EMIS_PROB
    new_emis_prob = {state: {s: 0 for s in emis_prob[states[0]]} for state in states}
    for i in states:
        denom = 0
        for t in range(len(obs)):
            denom += gamma[i][t]
        for j in emis_prob[states[0]]:
            numer = 0
            for t in range(len(obs)):
                if obs[t] == j:
                    numer += gamma[i][t]
            new_emis_prob[i][j] = numer / denom
    return new_start_prob, new_trans_prob, new_emis_prob


new_start_prob, new_trans_prob, new_emis_prob = baum_welch_algorithm(observations, states,
                                                                     alpha, beta, A_mod1,B)

# print('new_start_prob', new_start_prob, 'new_trans_prob', new_trans_prob, 'new_emis_prob', new_emis_prob)


def log_prob(scale_factor):
    return - sum(np.log(scale_factor))


# %%

def hmm(states, obs, start_prob, trans_prob, emis_prob):
    max_iters = 1000
    likelihoods = np.zeros(max_iters)
    old_log_prob = -1000000
    for i in range(max_iters):
        alpha, beta, scale_factor = forward_backward_procedure(obs, states, start_prob,
                                                               trans_prob, emis_prob)
        start_prob, trans_prob, emis_prob = baum_welch_algorithm(obs, states, alpha,
                                                                 beta, trans_prob, emis_prob)
        log_p = log_prob(scale_factor)
        likelihoods[i] = log_p
        if abs(log_p - old_log_prob) <= 0.0001:
            return start_prob, trans_prob, emis_prob, likelihoods[:(i + 1)]
        old_log_prob = log_p
    return start_prob, trans_prob, emis_prob, likelihoods


new_start_prob_1, new_trans_prob_1, new_emis_prob_1, likelihoods_1 = hmm(states, observations, pi, A_mod1, B)
new_start_prob_2, new_trans_prob_2, new_emis_prob_2, likelihoods_2 = hmm(states, observations, pi, A_mod2, B)

plt.plot(likelihoods_1, label='likelihood 1')
plt.plot(likelihoods_2, label='likelihood 2')
plt.legend()
plt.show()
