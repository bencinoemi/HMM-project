import numpy as np
class HmmBuilder:
    def __init__(self, obs, states, start_probability, transition_probability, emission_probability):
        self.obs = obs
        self.n_obs = len(obs)
        self.states = states
        self.n_states = len(states)
        self.start_prob = start_probability
        self.trans_prob = transition_probability
        self.emit_prob = emission_probability
        self.emit = self.emissions_symbols(self.obs)

    def emissions_symbols(self, obs):
        emit = []
        for i in obs:
            if i not in emit:
                emit.append(i)
        return np.asarray(emit)

    def forward_step_numpy(self):
        # initialize alpha, c0
        alpha = np.zeros((self.n_obs, self.n_states))
        scale_factor = np.zeros(self.n_obs)
        # compute alpha_0(i)

        alpha[0] = self.start_prob * self.emit_prob[:, np.where(self.emit == self.obs[0])[0][0]]
        # scaling alpha_0(i)
        scale_factor[0] = 1 / sum(alpha[0])

        alpha[0] = scale_factor[0] * alpha[0]
        # compute alpha_t(i)
        for t in range(1, self.n_obs):
            alpha[t] = alpha[t - 1].dot(self.trans_prob) * self.emit_prob[:, np.where(self.emit == self.obs[t])[0][0]]
            # scale alpha_t(i)
            scale_factor[t] = 1 / sum(alpha[t])
            alpha[t] = scale_factor[t] * alpha[t]
        return alpha, scale_factor

    def backward_step_numpy(self, scale_factor):
        # initialize beta_T scaled
        beta = np.zeros((self.n_obs, self.n_states))
        beta[self.n_obs - 1] = scale_factor[self.n_obs - 1]
        # compute beta_t(i)
        for t in range(self.n_obs - 2, -1, -1):
            beta[t] = self.trans_prob.dot(self.emit_prob[:, np.where(self.emit == self.obs[t + 1])[0][0]] * beta[t + 1])
            # scale beta_t(i)
            beta[t] = scale_factor[t] * beta[t]
        return beta

    def gamma_and_gamma_double(self, alpha, beta):
        # initialize gamma_t(i) and gamma_t(i, j)
        gamma_double = np.zeros((self.n_states, self.n_states, self.n_obs))

        for t in range(self.n_obs - 1):
            gamma_double[:, :, t] = alpha[t][:, np.newaxis] * self.trans_prob * \
                                    self.emit_prob[:, np.where(self.emit == self.obs[t + 1])[0][0]] * beta[t + 1]

        gamma = np.sum(gamma_double, axis=1)
        gamma[:, self.n_obs - 1] = alpha[self.n_obs - 1, :]
        return gamma, gamma_double

    def baum_welch_algorithm_numpy(self, alpha, beta):
        # COMPUTE GAMMA
        gamma, gamma_double = self.gamma_and_gamma_double(alpha, beta)

        # RE-ESTIMATE START_PROB
        new_start_prob = gamma[:, 0]

        # RE-ESTIMATE TRANS_PROB
        new_trans_prob = np.sum(gamma_double, axis=2) / np.sum(gamma, axis=1)[:, np.newaxis]

        # RE-ESTIMATE EMIT_PROB
        new_emit_prob = np.zeros((self.emit_prob.shape[0], self.emit_prob.shape[1]))

        denominator = np.sum(gamma, axis=1)
        for i in range(self.emit_prob.shape[1]):
            numerator = np.sum(gamma[:, np.where(self.obs == self.emit[i])[0]], axis=1)
            new_emit_prob[:, i] = numerator / denominator

        # if (abs(np.sum(new_start_prob) - 1.) > 0.1) or sum(abs(np.sum(new_trans_prob, axis=1) - 1.) > 0.1) != 0\
        #         or sum(abs(np.sum(new_emit_prob, axis=1) - 1.) > 0.1) != 0:
        #     raise ValueError
        # print(np.sum(new_start_prob), np.sum(new_trans_prob, axis=1), np.sum(new_emit_prob, axis=1))
        return new_start_prob, new_trans_prob, new_emit_prob

    def log_prob(self):
        return - np.log(sum(self.forward_step_numpy()[1]))

    def hmm_numpy(self):
        max_iter = 10000
        likelihoods = np.zeros(max_iter)
        old_log_prob = -1000000
        temp = HmmBuilder(self.obs, self.states, self.start_prob, self.trans_prob, self.emit_prob)
        for i in range(max_iter):
            alpha, scale = temp.forward_step_numpy()
            beta = temp.backward_step_numpy(scale)
            start_prob, trans_prob, emit_prob = temp.baum_welch_algorithm_numpy(alpha, beta)
            log_p = temp.log_prob()
            likelihoods[i] = log_p
            if abs(log_p - old_log_prob) <= 1e-5:
                return start_prob, trans_prob, emit_prob, likelihoods[:(i + 1)]
            old_log_prob = log_p
            temp = HmmBuilder(self.obs, self.states, start_prob, trans_prob, emit_prob)
        return start_prob, trans_prob, emit_prob, likelihoods

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.get_likelihood())
        plt.show()

    def plot_startprob(self):
        import matplotlib.pyplot as plt
        plt.plot(self.get_start_prob(), label='start_p')
        plt.plot(self.get_trans_prob(), label='trans_p')
        plt.plot(self.get_emis_prob(), label='emit_p')
        plt.legend()
        plt.show()

    def get_start_prob(self):
        start_prob, trans_prob, emit_prob, likelihoods = self.hmm_numpy()
        return start_prob

    def get_trans_prob(self):
        start_prob, trans_prob, emit_prob, likelihoods = self.hmm_numpy()
        return trans_prob

    def get_emis_prob(self):
        start_prob, trans_prob, emit_prob, likelihoods = self.hmm_numpy()
        return emit_prob

    def get_likelihood(self):
        start_prob, trans_prob, emit_prob, likelihoods = self.hmm_numpy()
        return likelihoods

    def viterbi(self):
        V1 = np.zeros((self.n_obs, self.n_states))
        V2 = np.empty([self.n_obs, self.n_states], dtype=object)

        start_prob, trans_prob, emit_prob, likelihoods = self.hmm_numpy()
        start_prob = np.log(start_prob + 0.00001)
        trans_prob = np.log(trans_prob + 0.00001)
        emit_prob = np.log(emit_prob + 0.00001)

        V1[0, :] = emit_prob[:, np.where(self.emit == self.obs[0])[0][0]] + start_prob
        # Run Viterbi when t > 0

        for t in range(1, self.n_obs):
            for st in range(self.n_states):
                max_tr_prob = V1[t - 1, 0] + trans_prob[0, st]
                prev_st_selected = self.states[0]
                for prev_st in range(1, self.n_states):
                    tr_prob = V1[t - 1, prev_st] + trans_prob[prev_st, st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = self.states[prev_st]
                max_prob = max_tr_prob + emit_prob[st, np.where(self.emit == self.obs[t])[0][0]]
                V1[t, st] = max_prob
                V2[t, st] = prev_st_selected

        max_prob = -np.inf
        opt = []
        best_st = ''
        # Get most probable state and its backtrack
        for i in range(self.n_states):
            if V1[-1, i] > max_prob:
                max_prob = V1[-1, i]
                best_st = self.states[i]

        opt.append(best_st)
        previous = best_st
        # Follow the backtrack till the first observation
        for t in range(self.n_obs - 2, -1, -1):
            opt.insert(0, V2[t + 1, np.where(np.asarray(self.states) == previous)[0][0]])
            previous = V2[t + 1, np.where(np.asarray(self.states) == previous)[0][0]]

        return np.asarray(opt), np.exp(max_prob)

    def viterbi_to_test(self, obs, states, start_prob, trans_prob, emit_prob):
        V1 = np.zeros((len(obs), len(states)))
        V2 = np.empty([len(obs), len(states)], dtype=object)

        start_prob = np.log(start_prob + 0.00001)
        trans_prob = np.log(trans_prob + 0.00001)
        emit_prob = np.log(emit_prob + 0.00001)

        emit = self.emissions_symbols(obs)
        V1[0, :] = emit_prob[:, np.where(emit == obs[0])[0][0]] + start_prob
        # Run Viterbi when t > 0

        for t in range(1, len(obs)):
            for st in range(len(states)):
                max_tr_prob = V1[t - 1, 0] + trans_prob[0, st]
                prev_st_selected = states[0]
                for prev_st in range(1, len(states)):
                    tr_prob = V1[t - 1, prev_st] + trans_prob[prev_st, st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = states[prev_st]
                max_prob = max_tr_prob + emit_prob[st, np.where(emit == obs[t])[0][0]]
                V1[t, st] = max_prob
                V2[t, st] = prev_st_selected

        max_prob = -np.inf
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

        return np.asarray(opt), np.exp(max_prob)


# hmm = HmmBuilder(observation, states, trans_p, start_p, emit_p)
# print(hmm.viterbi())