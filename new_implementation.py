import numpy as np
import pandas as pd

from itertools import product
from functools import reduce

class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs = probabilities.values()

        assert len(states) == len(probs), "The probabilities must match the states."

        assert len(states) == len(set(states)), "The states must be unique."

        assert abs(sum(probs) - 1.0) < 1e-7, "Probabilities must sum up to 1."
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), \
            "Probabilities must be numbers from [0, 1] interval."

        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x: probabilities[x], self.states))).reshape(1, -1)

    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size ** 2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))

    @classmethod
    def from_numpy(cls, array: np.ndarray, states: list):
        return cls(dict(zip(states, list(array))))

    @property
    def dict(self):
        return {k: v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]


class ProbabilityMatrix:
    def __init__(self, prob_vec_dict: dict):
        assert len(prob_vec_dict) > 1, \
            "The number of input probability vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1, \
            "All internal states of all the vectors must be equal."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())), \
            "All observables must be unique."

        self.states = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values = np.stack([prob_vec_dict[x].values for x in self.states]).squeeze()

    @classmethod
    def initialize(cls, states: list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables)) / (size ** 2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        aggr = [dict(zip(observables, rand[k, :])) for k in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array: np.ndarray, states: list, observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x))) for x in array]
        return cls(dict(zip(states, p_vecs)))

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def df(self):
        return pd.DataFrame(self.values,
                            columns=self.observables, index=self.states)

    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)


class HiddenMarkovChain:
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables

    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))

    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)

    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))

    def score(self, observations: list) -> float:
        def mul(x, y): return x * y

        score = 0
        all_chains = self._create_all_chains(len(observations))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))

            p_observations = list(map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]

            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score

class HiddenMarkovChain_FP(HiddenMarkovChain):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1)
                            @ self.T.values) * self.E[observations[t]].T
        return alphas

    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float(alphas[-1].sum())

class HiddenMarkovChain_Simulation(HiddenMarkovChain):
    def run(self, length: int) -> (list, list):
        assert length >= 0, "The chain needs to be a non-negative number."
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)

        prb = self.pi.values
        obs = prb @ self.E.values
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())

        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs = prb @ self.E.values
            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())

        return o_history, s_history


class HiddenMarkovChain_Uncover(HiddenMarkovChain_Simulation):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) @ self.T.values) \
                           * self.E[observations[t]].T
        return alphas

    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)
        return betas

    def uncover(self, observations: list) -> list:
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], maxargs))

class HiddenMarkovLayer(HiddenMarkovChain_Uncover):
    def _digammas(self, observations: list) -> np.ndarray:
        L, N = len(observations), len(self.states)
        digammas = np.zeros((L - 1, N, N))

        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.score(observations)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            digammas[t, :, :] = P1 * P2 / score
        return digammas


class HiddenMarkovModel:
    def __init__(self, hml: HiddenMarkovLayer):
        self.layer = hml
        self._score_init = 0
        self.score_history = []

    @classmethod
    def initialize(cls, states: list, observables: list):
        layer = HiddenMarkovLayer.initialize(states, observables)
        return cls(layer)

    def update(self, observations: list) -> float:
        alpha = self.layer._alphas(observations)
        beta = self.layer._betas(observations)
        digamma = self.layer._digammas(observations)
        score = alpha[-1].sum()
        gamma = alpha * beta / score

        L = len(alpha)
        obs_idx = [self.layer.observables.index(x) for x in observations]
        capture = np.zeros((L, len(self.layer.states), len(self.layer.observables)))
        for t in range(L):
            capture[t, :, obs_idx[t]] = 1.0

        pi = gamma[0]
        T = digamma.sum(axis=0) / gamma[:-1].sum(axis=0).reshape(-1, 1)
        E = (capture * gamma[:, :, np.newaxis]).sum(axis=0) / gamma.sum(axis=0).reshape(-1, 1)

        self.layer.pi = ProbabilityVector.from_numpy(pi, self.layer.states)
        self.layer.T = ProbabilityMatrix.from_numpy(T, self.layer.states, self.layer.states)
        self.layer.E = ProbabilityMatrix.from_numpy(E, self.layer.states, self.layer.observables)

        return score

    def train(self, observations: list, epochs: int, tol=None):
        self._score_init = 0
        self.score_history = (epochs + 1) * [0]
        early_stopping = isinstance(tol, (int, float))

        for epoch in range(1, epochs + 1):
            score = self.update(observations)
            print("Training... epoch = {} out of {}, score = {}.".format(epoch, epochs, score))
            if early_stopping and abs(self._score_init - score) / score < tol:
                print("Early stopping.")
                break
            self._score_init = score
            self.score_history[epoch] = score


all_possible_states = ['Toileting', 'Preparing breakfast', 'Bathing', 'Dressing', 'Grooming', 'Going out to work',
                       'Preparing lunch', 'Preparing a beverage', 'Washing dishes', 'Going out for shopping',
                       'Putting away groceries']
all_states_chains = ['Toileting', 'Toileting', 'Toileting', 'Toileting', 'Toileting', 'Toileting', 'Toileting',
                     'Preparing breakfast', 'Preparing breakfast', 'Preparing breakfast', 'Preparing breakfast',
                     'Bathing', 'Bathing', 'Bathing', 'Bathing', 'Dressing', 'Dressing', 'Dressing', 'Dressing',
                     'Dressing', 'Preparing breakfast', 'Dressing', 'Preparing breakfast', 'Preparing breakfast',
                     'Preparing breakfast', 'Preparing breakfast', 'Preparing breakfast', 'Preparing breakfast',
                     'Grooming', 'Preparing breakfast', 'Grooming', 'Grooming', 'Grooming', 'Grooming', 'Grooming',
                     'Going out to work', 'Preparing lunch', 'Preparing lunch', 'Preparing lunch', 'Preparing lunch',
                     'Preparing lunch', 'Preparing lunch', 'Preparing lunch', 'Preparing lunch', 'Preparing lunch',
                     'Preparing lunch', 'Preparing lunch', 'Preparing lunch', 'Preparing lunch', 'Preparing lunch',
                     'Preparing lunch', 'Preparing lunch', 'Preparing lunch', 'Preparing lunch', 'Preparing lunch',
                     'Preparing lunch', 'Preparing lunch', 'Preparing lunch', 'Preparing lunch', 'Preparing lunch',
                     'Preparing lunch', 'Preparing lunch', 'Grooming', 'Grooming', 'Grooming', 'Grooming', 'Grooming',
                     'Going out to work', 'Going out to work', 'Preparing a beverage', 'Preparing a beverage',
                     'Preparing a beverage', 'Preparing a beverage', 'Preparing a beverage', 'Toileting',
                     'Washing dishes', 'Washing dishes', 'Grooming', 'Grooming', 'Grooming', 'Grooming',
                     'Going out for shopping', 'Putting away groceries', 'Putting away groceries',
                     'Putting away groceries', 'Putting away groceries', 'Putting away groceries',
                     'Putting away groceries', 'Putting away groceries', 'Dressing', 'Dressing', 'Dressing',
                     'Dressing', 'Dressing', 'Dressing']

poss_sens = ['67', '101', '57', '58', '82', '71', '80', '143', '55', '54', '93', '72', '75', '84', '73', '70', '135',
             '91', '88', '68', '140', '137', '94', '95', '53', '62',  '66',  '92', '130', '104', '105',  '78']
chain_length = len(all_states_chains)

array_pi = np.array((0.08421053, 0.12631579, 0.04210526, 0.12631579, 0.15789474, 0.03157895,
                     0.27368421, 0.05263158, 0.02105263, 0.01052632, 0.07368421))
d_pi = {}
for i in range(len(all_possible_states)):
    d_pi[all_possible_states[i]] = array_pi[i]
pi = ProbabilityVector(d_pi)
# print(pi)

array_A = np.array(((7.50000000e-01, 1.25000000e-01, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.25000000e-01, 1.00000000e-12, 1.00000000e-12),
                    (1.00000000e-12, 6.66666667e-01, 8.33333333e-02, 8.33333333e-02,  1.66666667e-01, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12),
                    (1.00000000e-12, 1.00000000e-12, 7.50000000e-01, 2.50000000e-01, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12),
                    (1.00000000e-12, 1.81818182e-01, 1.00000000e-12, 8.18181818e-01, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12),
                    (1.00000000e-12, 6.66666667e-02, 1.00000000e-12, 1.00000000e-12, 7.33333333e-01, 1.33333333e-01,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 6.66666667e-02, 1.00000000e-12),
                    (1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 3.33333333e-01,
                     3.33333333e-01, 3.33333333e-01, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12),
                    (1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 3.84615385e-02, 1.00000000e-12,
                     9.61538462e-01, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12),
                    (2.00000000e-01, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 8.00000000e-01, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12),
                    (1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 5.00000000e-01, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 5.00000000e-01, 1.00000000e-12, 1.00000000e-12),
                    (1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e+00),
                    (1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.42857143e-01, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 8.57142857e-01)))

d_A = {}
for i in range(len(all_possible_states)):
    d_ai = {}
    for j in range(len(all_possible_states)):
        # print(array_A[i, j])
        d_ai[all_possible_states[j]] = array_A[i, j]
    d_A[all_possible_states[i]] = ProbabilityVector(d_ai)

A = ProbabilityMatrix(d_A)
# print(A)

array_B = np.array(((0.25,  0.125,  0.125,  0.125, 0.125, 0.125,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 0.125, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12),
                    (1.00000000e-12, 0.        , 0.        , 0.08333333, 0.        ,
                     0.        , 0.08333333, 0.08333333, 0.08333333, 0.08333333,
                     0.        , 0.08333333, 0.        , 0.08333333, 0.16666667,
                     0.08333333, 0.08333333, 0.08333333, 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0., 0.),
                    (0.25      , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.25      ,
                     0.25      , 0.25      , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.),
                    (1.00000000e-12, 1.00000000e-12, 8.33333333e-02, 1.00000000e-12,
                     2.50000000e-01, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12,
                     2.50000000e-01, 8.33333333e-02, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 1.00000000e-12,
                     8.33333333e-02, 8.33333333e-02, 1.00000000e-12, 1.00000000e-12,
                     1.00000000e-12, 1.00000000e-12, 1.00000000e-12, 8.33333333e-02,
                     1.00000000e-12, 8.33333333e-02, 1.00000000e-12, 1.00000000e-12),
                    (0.        , 0.13333333, 0.26666667, 0.13333333, 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.06666667,
                     0.06666667, 0.        , 0.        , 0.13333333, 0.13333333,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.06666667, 0.        , 0.        , 0.        ,
                     0.        , 0.        ),
                    (0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.33333333, 0.        , 0.        ,
                     0.66666667, 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        ),
                    (0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.19230769, 0.        ,
                     0.        , 0.03846154, 0.        , 0.19230769, 0.03846154,
                     0.03846154, 0.        , 0.11538462, 0.        , 0.        ,
                     0.03846154, 0.19230769, 0.03846154, 0.03846154, 0.03846154,
                     0.03846154, 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.),
                    (0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.2       , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.4       , 0.        , 0.        ,
                     0.2       , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.2       , 0.        , 0.        ,
                     0.        , 0.),
                    (0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.5       , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.5       , 0.        , 0.        , 0.        ,
                     0.        , 0.),
                    (0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     1.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.       ),
                    (0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.        , 0.        , 0.14285714,
                     0.        , 0.        , 0.        , 0.        , 0.        ,
                     0.14285714, 0.14285714, 0.        , 0.        , 0.        ,
                     0.        , 0.        , 0.14285714, 0.        , 0.14285714,
                     0.14285714, 0.14285714)))

d_B = {}
for i in range(len(all_possible_states)):
    d_bi = {}
    for j in range(len(poss_sens)):
        # print(array_A[i, j])
        d_bi[poss_sens[j]] = array_B[i, j]
    d_B[all_possible_states[i]] = ProbabilityVector(d_bi)

B = ProbabilityMatrix(d_B)
# print(B)

hmc = HiddenMarkovChain(A, B, pi)
observations = ['67', '101', '57', '58', '67', '82', '71', '80', '143', '55', '54', '54', '93', '72', '67', '57', '82',
                '75', '82', '75', '84', '84', '72', '73', '70', '135', '91', '73', '73', '58', '58', '88', '68', '57',
                '57', '140', '140', '137', '73', '55', '94', '91', '84', '55', '84', '95', '55', '91', '137', '72',
                '137', '55', '84', '84', '137', '55', '53', '70', '91', '137', '84', '62', '70', '66', '88', '68', '57',
                '91', '140', '140', '92', '91', '80', '91', '130', '70', '66', '58', '57', '101', '101', '140', '140',
                '104', '92', '105', '73', '137', '78', '137', '82', '75', '92', '140', '104']

print("Score for {} is {:f}.".format(observations, hmc.score(observations)))


'''all_possible_observations = {'1S', '2M', '3L'}
chain_length = 3  # any int > 0
all_observation_chains = list(product(*(all_possible_observations,) * chain_length))
all_possible_scores = list(map(lambda obs: hmc.score(obs), all_observation_chains))
print("All possible scores added: {}.".format(sum(all_possible_scores)))


hmc_s = HiddenMarkovChain_Simulation(A, B, pi)

stats = {}
for length in np.logspace(1, 5, 40).astype(int):
    observation_hist, states_hist = hmc_s.run(length)
    stats[length] = pd.DataFrame({'observations': observation_hist,
                                  'states': states_hist}).applymap(lambda x: int(x[0]))

S = np.array(list(map(lambda x: x['states'].value_counts().to_numpy() / len(x), stats.values())))

plt.semilogx(np.logspace(1, 5, 40).astype(int), S)
plt.xlabel('Chain length T')
plt.ylabel('Probability')
plt.title('Converging probabilities.')
plt.legend(['1H', '2C'])
plt.show()


np.random.seed(42)
'''


'''
a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})
b1 = ProbabilityVector({'1S': 0.1, '2M': 0.4, '3L': 0.5})
b2 = ProbabilityVector({'1S': 0.7, '2M': 0.2, '3L': 0.1})
A = ProbabilityMatrix({'1H': a1, '2C': a2})
B = ProbabilityMatrix({'1H': b1, '2C': b2})
pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

hmc = HiddenMarkovChain_Uncover(A, B, pi)

observed_sequence, latent_sequence = hmc.run(5)
uncovered_sequence = hmc.uncover(observed_sequence)

all_possible_states = {'1H', '2C'}
chain_length = 6 # any int > 0
all_states_chains = list(product(*(all_possible_states,) * chain_length))

df = pd.DataFrame(all_states_chains)
dfp = pd.DataFrame()

for i in range(chain_length):
    dfp['p' + str(i)] = df.apply(lambda x: hmc.E.df.loc[x[i], observed_sequence[i]], axis=1)

scores = dfp.sum(axis=1).sort_values(ascending=False)
df = df.iloc[scores.index]
df['score'] = scores
# print(df.head(10).reset_index())

dfc = df.copy().reset_index()
for i in range(chain_length):
    dfc = dfc[dfc[i] == latent_sequence[i]]

# print(dfc)

np.random.seed(42)

observations = ['67', '101', '57', '58', '67', '82', '71', '80', '143', '55', '54', '54', '93', '72', '67', '57', '82',
                '75', '82', '75', '84', '84', '72', '73', '70', '135', '91', '73', '73', '58', '58', '88', '68', '57',
                '57', '140', '140', '137', '73', '55', '94', '91', '84', '55', '84', '95', '55', '91', '137', '72',
                '137', '55', '84', '84', '137', '55', '53', '70', '91', '137', '84', '62', '70', '66', '88', '68', '57',
                '91', '140', '140', '92', '91', '80', '91', '130', '70', '66', '58', '57', '101', '101', '140', '140',
                '104', '92', '105', '73', '137', '78', '137', '82', '75', '92', '140', '104']

states = ['Toileting', 'Preparing breakfast', 'Bathing', 'Dressing', 'Grooming', 'Going out to work', 'Preparing lunch',
          'Preparing a beverage', 'Washing dishes', 'Going out for shopping', 'Putting away groceries']
observables = ['67', '101', '57', '58', '82', '71', '80', '143', '55', '54', '93', '72', '75', '84', '73', '70',
               '135', '91', '88', '68', '140', '137', '94', '95', '53', '62', '66', '92', '130', '104', '105', '78']

observations = ['3L', '2M', '1S', '3L', '3L', '3L']

states = ['1H', '2C']
observables = ['1S', '2M', '3L']
''''''
hml = HiddenMarkovLayer.initialize(states, observables)
hmm = HiddenMarkovModel(hml, tol=0.0001)

hmm.train(observations, 25)

RUNS = 100000
T = 5

chains = RUNS * [0]
for i in range(len(chains)):
    chain = hmm.layer.run(T)[0]
    chains[i] = '-'.join(chain)

df = pd.DataFrame(pd.Series(chains).value_counts(), columns=['counts']).reset_index().rename(columns={'index': 'chain'})
df = pd.merge(df, df['chain'].str.split('-', expand=True), left_index=True, right_index=True)

s = []
for i in range(T + 1):
    s.append(df.apply(lambda x: x[i] == observations[i], axis=1))

df['matched'] = pd.concat(s, axis=1).sum(axis=1)
df['counts'] = df['counts'] / RUNS * 100
df = df.drop(columns=['chain'])
print(df.head(30))


hml_rand = HiddenMarkovLayer.initialize(states, observables)
hmm_rand = HiddenMarkovModel(hml_rand)

RUNS = 100000
T = 5

chains_rand = RUNS * [0]
for i in range(len(chains_rand)):
    chain_rand = hmm_rand.layer.run(T)[0]
    chains_rand[i] = '-'.join(chain_rand)

df2 = pd.DataFrame(pd.Series(chains_rand).value_counts(), columns=['counts']).reset_index().rename(columns={'index': 'chain'})
df2 = pd.merge(df2, df2['chain'].str.split('-', expand=True), left_index=True, right_index=True)

s = []
for i in range(T + 1):
    s.append(df2.apply(lambda x: x[i] == observations[i], axis=1))

df2['matched'] = pd.concat(s, axis=1).sum(axis=1)
df2['counts'] = df2['counts'] / RUNS * 100
df2 = df2.drop(columns=['chain'])

fig, ax = plt.subplots(1, 1, figsize=(14, 6))

ax.plot(df['matched'], 'g:')
ax.plot(df2['matched'], 'k:')

ax.set_xlabel('Ordered index')
ax.set_ylabel('Matching observations')
ax.set_title('Verification on a 6-observation chain.')

ax2 = ax.twinx()
ax2.plot(df['counts'], 'r', lw=3)
ax2.plot(df2['counts'], 'k', lw=3)
ax2.set_ylabel('Frequency of occurrence [%]')

ax.legend(['trained', 'initialized'])
ax2.legend(['trained', 'initialized'])

plt.grid()
plt.show()'''

