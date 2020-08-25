#
import numpy as np
import matplotlib.pyplot as plt
from hmm_class import HmmBuilder

"""
states = np.array(('Healthy', 'Fever'), 'str')
observations = np.array((0, 1, 2, 2, 1, 0, 0,1,2,1,0), 'int')
start_probs = np.array((0.6, 0.4), 'float')
trans_probs = np.array(((0.69, 0.3), (0.4, 0.59)), 'float')
emis_probs = np.array(((0.5, 0.4, 0.1), (0.1, 0.3, 0.6)), 'float')
"""
"""states = ('Healthy', 'Fever')
end_state = 'E'
observations = ['normal', 'cold', 'dizzy', 'normal']
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
transition_probability = {
    'Healthy': {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01]
    'Fever': {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01}}
emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}}"""

states = ('first', 'second', 'third')
observations = np.array(('two', 'two', 'one', 'two', 'one', 'one', 'two', 'one', 'one', 'two'))
A_mod1 = np.array(((0.6, 0.1, 0.3),
                   (0.566, 0.122, 0.312),
                   (0.1, 0.4, 0.5)))

A_mod2 = np.array(((0.6, 0.2, 0.2),
                   (0.2, 0.4, 0.4),
                   (0.1, 0.45, 0.45)))

B = np.array(((0.5, 0.5),
              (0.75, 0.25),
              (0.25, 0.75)))

pi = np.array((0.333, 0.333, 0.333))

# prova = HmmBuilder(observations, states, A_mod1, pi, B)
# print(prova.viterbi())

# forward_backward_procedure
def forward_step_numpy(n_obs, n_states, start_prob, emis_prob, emis, obs, transition_prob):
    # initialize alpha, c0
    alpha = np.zeros((n_obs, n_states))
    scale_factor = np.zeros(n_obs)
    # compute alpha_0(i)
    alpha[0] = start_prob * emis_prob[:, np.where(emis == obs[0])[0][0]]
    # scaling alpha_0(i)
    scale_factor[0] = 1 / sum(alpha[0])
    alpha[0] = scale_factor[0] * alpha[0]

    # compute alpha_t(i)
    for t in range(1, n_obs):
        alpha[t] = alpha[t - 1].dot(transition_prob) * emis_prob[:, np.where(emis == obs[t])[0][0]]
        # scale alpha_t(i)
        scale_factor[t] = 1 / sum(alpha[t])
        alpha[t] = scale_factor[t] * alpha[t]
    return alpha, scale_factor


def backword_step_numpy(n_obs, n_states, transition_prob, emis_prob, emis, obs, scale_factor):
    # initialize beta_T scaled
    beta = np.zeros((n_obs, n_states))
    beta[n_obs - 1] = scale_factor[n_obs - 1]
    # compute beta_t(i)
    for t in range(n_obs - 2, -1, -1):
        # beta[t] = beta[t + 1].dot(transition_prob) * emis_prob[:, np.where(emis == obs[t + 1])[0][0]]
        beta[t] = transition_prob.dot(emis_prob[:, np.where(emis == obs[t + 1])[0][0]] * beta[t + 1])
        # scale beta_t(i)
        beta[t] = scale_factor[t] * beta[t]
    return beta


def forward_backward_procedure_numpy(obs, states, start_prob, transition_prob, emis_prob):
    n_obs = len(obs)
    n_states = len(states)
    emis = np.unique(obs)

    # FARWARD STEP
    alpha, scale_factor = forward_step_numpy(n_obs, n_states, start_prob, emis_prob, emis, obs, transition_prob)

    # BACKWARD STEP
    beta = backword_step_numpy(n_obs, n_states, transition_prob, emis_prob, emis, obs, scale_factor)

    return alpha, beta, scale_factor


# alpha, beta, scale_factor = forward_backward_procedure_numpy(observations, states, start_probs,
#                                                               trans_probs, emis_probs)
# alpha, beta, scale_factor = forward_backward_procedure_numpy(observations, states, pi, A_mod1, B)

# print('alpha', alpha, 'beta', beta, scale_factor)

# new_start_prob, new_trans_prob, new_emis_prob = baum_welch_algorithm_numpy(observations, states,
#                                                                      alpha, beta, trans_probs,
#                                                                      emis_probs)
# print('new_start_prob', new_start_prob, 'new_trans_prob', new_trans_prob, 'new_emis_prob', new_emis_prob)

def gamma_and_gamma_double(obs, states, alpha, beta, trans_prob, emis_prob):
    # initialize gamma_t(i) and gamma_t(i, j)
    emis = np.unique(obs)
    gamma_double = np.zeros((len(states), len(states), len(obs)))

    for t in range(len(obs) - 1):
        gamma_double[:, :, t] = alpha[t][:, np.newaxis] * trans_prob * emis_prob[:,
                                                                       np.where(emis == obs[t + 1])[0][0]] * beta[t + 1]

    gamma = np.sum(gamma_double, axis=1)
    gamma[:, len(obs) - 1] = alpha[len(obs) - 1, :]
    return gamma, gamma_double


# g, g_d = gamma_and_gamma_double(observations, states, alpha, beta, A_mod1, B)
# print('gamma', g, 'gamma_double', g_d)

def baum_welch_algorithm_numpy(obs, states, alpha, beta, trans_prob, emis_prob):
    emis = np.unique(obs)

    # COMPUTE GAMMA
    gamma, gamma_double = gamma_and_gamma_double(obs, states, alpha, beta, trans_prob, emis_prob)

    # RE-ESTIMATE START_PROB
    new_start_prob = gamma[:, 0]

    # RE-ESTIMATE TRANS_PROB
    new_trans_prob = np.sum(gamma_double, axis=2) / np.sum(gamma[:, :9], axis=1)[:, np.newaxis]

    # RE-ESTIMATE EMIS_PROB
    new_emis_prob = np.zeros((emis_prob.shape[0], emis_prob.shape[1]))

    denominator = np.sum(gamma, axis=1)
    for i in range(emis_prob.shape[1]):
        numerator = np.sum(gamma[:, np.where(obs == emis[i])[0]], axis=1)
        new_emis_prob[:, i] = numerator / denominator

    return new_start_prob, new_trans_prob, new_emis_prob


# new_start_prob, new_trans_prob, new_emis_prob = baum_welch_algorithm_numpy(observations, states, alpha, beta, A_mod1, B)
# print('new start', new_start_prob, 'new trans', new_trans_prob, 'new emis', new_emis_prob)

def log_prob(scale_factor):
    return - np.log(sum(scale_factor))


# l = log_prob(scale_factor)
# print(l)
# %%

def hmm_numpy(states, obs, start_prob, trans_prob, emis_prob):
    max_iters = 10000
    likelihoods = np.zeros(max_iters)
    old_log_prob = -1000000
    for i in range(max_iters):

        alpha, beta, scale_factor = forward_backward_procedure_numpy(obs, states, start_prob, trans_prob, emis_prob)
        start_prob, trans_prob, emis_prob = baum_welch_algorithm_numpy(obs, states, alpha, beta, trans_prob, emis_prob)

        log_p = log_prob(scale_factor)
        # print(- np.log(sum(scale_factor)))
        likelihoods[i] = log_p
        if abs(log_p - old_log_prob) <= 0.0001:
            return start_prob, trans_prob, emis_prob, likelihoods[:(i + 1)]
        old_log_prob = log_p
    return start_prob, trans_prob, emis_prob, likelihoods


# new_start_prob, new_trans_prob, new_emis_prob, likelihoods = hmm_1(states, observations, start_probs,
#                                                               trans_probs, emis_probs)
new_start_prob_1, new_trans_prob_1, new_emis_prob_1, likelihoods_1 = hmm_numpy(states, observations, pi, A_mod1, B)
new_start_prob_2, new_trans_prob_2, new_emis_prob_2, likelihoods_2 = hmm_numpy(states, observations, pi, A_mod2, B)

# print(likelihoods_1)
# plt.plot(likelihoods)
plt.plot(likelihoods_1, label='mod_1')
plt.plot(likelihoods_2, label='mod_2')
plt.legend()
plt.show()


def unique_observation(obs):
    emis = []
    for i in obs:
        if i not in emis:
            emis.append(i)
    return np.asarray(emis)


# TODO: implementa di nuovo Viterbi perchè c'è qualcosa che non va
'''
function
VITERBI{(O, S,\Pi, Y, A, B):X}
for each state{i = 1, 2,.., K} do
    T_{1}[i, 1] <- pi{i} * B_{iy_{1}
    T_{2}[i, 1] <- 0}
for each observation j = 2, 3,.., T do
    for each state i = 1, 2,.., K do
        T_{1}[i, j] <- \max_{k} {(T_{1}[k, j-1] * A_{ki} * B_{iy_{j}})}}
        T_{2}[i, j] <- \arg \max_{k} {(T_{1}[k, j-1] * A_{ki} * B_{iy_{j}})}}
z_{T} <- \arg \max_{k} {(T_{1}[k, T])}}z_{T}\gets \arg \max_{k}{(T_{1}[k, T])}
x_{T} <-s_{z_{T}}}
for j = T, T - 1,.., 2 do
    z_{j - 1} <-T_{2}[z_{j}, j]}
    x_{j - 1} <-s_z_{j - 1}}}
return X'''

def other_viterbi(obs, states, start_p, trans_p, emit_p):
    t1 = np.zeros((len(obs), len(states)))
    t2 = np.zeros((len(obs), len(states)))
    emit = unique_observation(obs)
    t1[:, 0] = emit_p[:, np.where(emit == obs[0])[0][0]].dot(trans_p)
    return None



def viterbi(obs, states, start_p, trans_p, emit_p):
    emis = unique_observation(obs)
    V1 = np.zeros((len(obs), len(states)))
    V2 = np.empty([len(obs), len(states)], dtype=object)
    emit_p = np.log(emit_p)
    start_p = np.log(start_p)
    trans_p = np.log(trans_p)

    V1[0, :] = emit_p[:, np.where(emis == obs[0])[0][0]] + start_p
    # Run Viterbi when t > 0

    for t in range(1, len(obs)):
        for st in range(len(states)):
            max_tr_prob = V1[t - 1, 0] + trans_p[0, st]
            prev_st_selected = states[0]
            for prev_st in range(1, len(states)):
                tr_prob = V1[t - 1, prev_st] + trans_p[prev_st, st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = states[prev_st]
            max_prob = max_tr_prob + emit_p[st, np.where(emis == obs[t])[0][0]]
            V1[t, st] = max_prob
            V2[t, st] = prev_st_selected

    max_prob = - np.inf
    opt = []
    best_st = ''
    # Get most probable state and its backtrack
    for i in range(len(states)):
        if V1[-1, i] > max_prob:
            max_prob = V1[-1, i]
            best_st = states[i]
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(obs) - 2, -1, -1):
        opt.insert(0, V2[t + 1, np.where(np.asarray(states) == previous)[0][0]])
        previous = V2[t + 1, np.where(np.asarray(states) == previous)[0][0]]

    return opt, max_prob


pred, prob = viterbi(observations, states, new_start_prob_1, new_trans_prob_1, new_emis_prob_1)
print(pred)


'''
import numpy as np

'''
observation = np.array(("normal", "cold", "dizzy"))
states = ("Healthy", "Fever")
start_p = np.array((0.6, 0.4))
trans_p = np.array(((0.7, 0.3),
                    (0.4, 0.6)))

emit_p = np.array(((0.5, 0.4, 0.1),
                   (0.1, 0.3, 0.6)))


obs = ("normal", "cold", "dizzy")
states_d = ("Healthy", "Fever")
start_p_d = {"Healthy": 0.6, "Fever": 0.4}
trans_p_d = {
    "Healthy": {"Healthy": 0.7, "Fever": 0.3},
    "Fever": {"Healthy": 0.4, "Fever": 0.6},
}
emit_p_d = {
    "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
    "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6},
}

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
    opt = []
    max_prob = 0.0
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)








def unique_observation(obs):
    emis = []
    for i in obs:
        if i not in emis:
            emis.append(i)
    return np.asarray(emis)

def viterbi(obs, states,  start_prob, trans_prob, emis_prob):
    n_obs = len(obs)
    n_states = len(states)
    emis = unique_observation(obs)
    trellis = np.zeros((n_states, n_obs))
    backpointer = np.zeros((n_states, n_obs), 'int')
    i_star = np.zeros(n_obs)

    # initialization step
    trellis[:, 0] = emis_prob[:, np.where(emis == obs[0])[0][0]].dot(trans_prob)

    # recursion step
    for t in range(1, n_obs):
        trellis[:, t] = max(trellis[:, (t-1)].dot(trans_prob)) * emis_prob[:, np.where(emis == obs[t])[0][0]]
        backpointer[:, t] = np.argmax(trans_prob * trellis[:, t-1], axis=0)

    # termination
    P = max(trellis[:, (n_obs-1)])
    i_star[n_obs-1] = np.where(max(trellis[:, (n_obs-1)]))[0][0]

    # state sequence betracking
    for t in range(n_obs-1, -1, -1):
        i_star[t] = backpointer[t+1]


def viterbi(obs, states, start_p, trans_p, emit_p):
    emis = unique_observation(obs)
    V1 = np.zeros((len(obs), len(states)))
    V2 = np.empty([len(obs), len(states)], dtype=object)

    V1[0, :] = start_p * emit_p[:, np.where(emis == obs[0])[0][0]]
    # Run Viterbi when t > 0

    for t in range(1, len(obs)):
        for st in range(len(states)):
            max_tr_prob = V1[t - 1, 0] * trans_p[0, st]
            prev_st_selected = states[0]
            for prev_st in range(1, len(states)):
                tr_prob = V1[t - 1, prev_st] * trans_p[prev_st, st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = states[prev_st]
            max_prob = max_tr_prob * emit_p[st, np.where(emis == obs[t])[0][0]]
            V1[t, st] = max_prob
            V2[t, st] = prev_st_selected


    max_prob = 0.0

    # Get most probable state and its backtrack
    for i in range(len(states)):
        if V1[-1, i] > max_prob:
            max_prob = V1[-1, i]
            best_st = states[i]
    opt = [best_st]
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(obs) - 2, -1, -1):
        opt.insert(0, V2[t + 1, np.where(np.asarray(states) == previous)][0][0])
        previous = V2[t + 1, np.where(np.asarray(states) == previous)[0][0]]

    return opt, max_prob
