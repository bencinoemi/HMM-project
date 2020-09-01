import pandas as pd
from datetime import datetime
from hmm_class import *

sensors = pd.read_csv('/home/noe/Università/in_corso/Machine Learning/progetto/dataset/subject1/sensors.csv',
                      header=None, names=['Sensors_code', 'Location', 'Sensors_name'])

activity_list = pd.read_csv('/home/noe/Università/in_corso/Machine Learning/progetto/dataset/subject1/Activities.csv')

data_1 = pd.read_csv('/home/noe/Università/in_corso/Machine Learning/progetto/dataset/subject1/activity_sub1.csv',
                   header=None, names=['Action', 'Date', 'Activity_start', 'Activity_end', 'Sensors_code',
                                       'Sensors_name', 'start_time', 'end_time'])
data_2 = pd.read_csv('/home/noe/Università/in_corso/Machine Learning/progetto/dataset/subject2/activity_sub2.csv',
                   header=None, names=['Action', 'Date', 'Activity_start', 'Activity_end', 'Sensors_code',
                                       'Sensors_name', 'start_time', 'end_time'])

activities = np.asarray(activity_list['Subcategory'])
# print(activities)

# data[1] = pd.to_datetime(data[1]).dt.date
# print(data[1].unique())
# days = data[1].unique()
print(data_2['Date'].unique())
data_1['Date'] = pd.to_datetime(data_1['Date'], format='%m/%d/%Y')
data_1['start_time'] = pd.to_datetime(data_1['start_time'], format="%H:%M:%S")
data_1['end_time'] = pd.to_datetime(data_1['end_time'], format="%H:%M:%S")

df_1 = pd.merge(data_1, sensors)
df_1 = df_1.sort_values(['Date', 'start_time'])

dates_1 = df_1.Date.unique()
times_1 = df_1.start_time.unique()

data_2['Date'].unique()
# data_2['Date'] = pd.to_datetime(data_2['Date'], format='%m/%d/%Y')


data_2['start_time'] = [datetime.strptime(x, "%H:%M:%S").time() for x in data_2['start_time']]
data_2['end_time'] = [datetime.strptime(x, "%H:%M:%S").time() for x in data_2['end_time']]
df_2 = pd.merge(data_2, sensors)
df_2 = df_2.sort_values(['Date', 'start_time'])

dates_2 = df_2.Date.unique()

times_2 = df_2.start_time.unique()

def unique_observation(seq):
    unique_seq = []
    for i in seq:
        if i not in unique_seq:
            unique_seq.append(i)
    return np.asarray(unique_seq)

def hyper_parameters(hidden_states, emit):
    pi = np.random.dirichlet(np.ones(len(hidden_states)))

    A = np.random.dirichlet(np.ones(len(hidden_states)), size=len(hidden_states))

    B = np.random.dirichlet(np.ones(len(emit)), size=len(hidden_states))
    return pi, A, B


def hyper_parameters_equals(hidden_states, emit):
    pi = np.array(np.repeat(1/len(hidden_states), len(hidden_states)))

    A = np.repeat((np.repeat(1/len(hidden_states), len(hidden_states))), len(hidden_states))
    A.resize(len(hidden_states), len(hidden_states))

    B = np.repeat((np.repeat(1/len(emit), len(emit))), len(hidden_states))
    B.resize(len(hidden_states), len(emit))

    return pi, A, B

def dict_creator(true_seq, h_states):
    dct = {}
    for s in h_states:
        dct[s] = [0, sum(np.asarray(true_seq) == s)]
    return dct

'''
for day in range(len(dates)):
    sensors_obs = np.asarray(df.loc[df.Date == dates[day]].Sensors_code)
    true_seq_activity = np.asarray(df.loc[df.Date == dates[day]].Action)
    states = unique_observation(true_seq_activity)
    emits = unique_observation(sensors_obs)
    start_p, trans_p, emit_p = hyper_parameters(states, emits)
    hmm_homemade = HmmBuilder(sensors_obs, states, start_p, trans_p, emit_p)

    start_p, trans_p, emit_p, lik = hmm_homemade.hmm_numpy()
    pred_seq = hmm_homemade.viterbi()[0]
    print(pred_seq)
    d = dict_creator(true_seq_activity, states)

    for i in range(len(pred_seq)):
        if pred_seq[i] == true_seq_activity[i]:
            d[pred_seq[i]] = [(d[pred_seq[i]][0] + 1) / d[pred_seq[i]][1],  d[pred_seq[i]][1]]
    print(d)

'''

'''
obs = []
true_seq_activity = []
for day in range(len(dates)):
    obs.extend(list(df.loc[df.Date == dates[day]].Sensors_code.values))
    true_seq_activity.extend(list(df.loc[df.Date == dates[day]].Action.values))

states = unique_observation(true_seq_activity)

emits = unique_observation(np.asarray(obs))
start_p, trans_p, emit_p = hyper_parameters(states, emits)

hmm_homemade = HmmBuilder(np.asarray(obs), states, start_p, trans_p, emit_p)
# alpha, scale = hmm_homemade.forward_step_numpy()
# beta = hmm_homemade.backward_step_numpy(scale)
# print(hmm_homemade.hmm_numpy())
# print(hmm_homemade.viterbi())
d = dict_creator(np.asarray(true_seq_activity), np.asarray(states))

pred_seq = hmm_homemade.viterbi()[0]
# print(sum(pred_seq == true_seq_activity) / len(true_seq_activity))
# print(true_seq_activity)

for i in range(len(pred_seq)):
    if pred_seq[i] == true_seq_activity[i]:
        d[pred_seq[i]] = [(d[pred_seq[i]][0] + 1) / d[pred_seq[i]][1], d[pred_seq[i]][1]]
print(hmm_homemade.plot(hmm_homemade.get_likelihood()))
'''
# #### TRY 26/08/2020
# initialization of the parameters in a peculiar way

#

def init_start_prob(unique_states, seq):
    '''
    Every prior prob i is:
    #of state i observed / #obs
    '''
    pi = []
    for s in unique_states:
        pi.append(sum(np.asarray(seq) == s))
    return np.asarray(pi) / len(seq)

def init_trans_prob(seq, unique_states):
    '''
    Every trans prob i, j is:
    #number of times the state i is followed by state j / #of times the state i appears
    '''
    a = np.zeros((len(unique_states), len(unique_states)))
    for s in range(len(unique_states)):
        for st in range(len(unique_states)):
            for t in range(len(seq) - 1):
                if seq[t] == unique_states[s] and seq[t + 1] == unique_states[st]:
                    a[s, st] += 1
        if seq[-1] == unique_states[s]:
            den = sum(np.asarray(seq) == unique_states[s]) - 1
        else:
            den = sum(np.asarray(seq) == unique_states[s])
        a[s] = a[s] / den
    return a

# every emit prob i, k is: #number of times the sensor k is observed in state i/ #obs
def init_emit_prob(emits, states, obs, true_seq_activity):
    b = np.zeros((len(states), len(emits)))
    for s in range(len(states)):
        for t in range(len(obs)):
            if true_seq_activity[t] == states[s]:
                for e in range(len(emits)):
                    if obs[t] == emits[e]:
                        b[s, e] += 1
        b[s] = b[s] / sum(np.asarray(true_seq_activity) == states[s])

    return b


# obs = []
# true_seq_activity = []

# for day in range(len(dates)):
#     obs.extend(list(df.loc[df.Date == dates[day]].Sensors_code.values))
#     true_seq_activity.extend(list(df.loc[df.Date == dates[day]].Action.values))

c = 0
for day in range(len(dates_1)):
    obs = list(df_1.loc[df_1.Date == dates_1[day]].Sensors_code.values)
    true_seq_activity = list(df_1.loc[df_1.Date == dates_1[day]].Action.values)
    states = unique_observation(true_seq_activity)

    emits = unique_observation(np.asarray(obs))
    start_p = init_start_prob(states, true_seq_activity) + 1e-12
    trans_p = init_trans_prob(true_seq_activity, states) + 1e-12
    emit_p = init_emit_prob(emits, states, obs, true_seq_activity) + 1e-12

    hmm_homemade = HmmBuilder(np.asarray(obs), states, start_p, trans_p, emit_p)

    d = dict_creator(np.asarray(true_seq_activity), np.asarray(states))

    pred_seq = hmm_homemade.viterbi()[0]

    for i in range(len(pred_seq)):
        if pred_seq[i] == true_seq_activity[i]:
            d[pred_seq[i]] = [(d[pred_seq[i]][0] + 1), d[pred_seq[i]][1]]
            c += 1

    print('subject_1', d)
    print( true_seq_activity, pred_seq)

print('subject_1', c / data_1.shape[0])


for day in range(len(dates_2)):
    obs = list(df_2.loc[df_2.Date == dates_2[day]].Sensors_code.values)
    true_seq_activity = list(df_2.loc[df_2.Date == dates_2[day]].Action.values)
    states = unique_observation(true_seq_activity)

    emits = unique_observation(np.asarray(obs))
    start_p = init_start_prob(states, true_seq_activity) + 1e-12
    trans_p = init_trans_prob(true_seq_activity, states) + 1e-12
    emit_p = init_emit_prob(emits, states, obs, true_seq_activity) + 1e-12

    hmm_homemade = HmmBuilder(np.asarray(obs), states, start_p, trans_p, emit_p)

    d = dict_creator(np.asarray(true_seq_activity), np.asarray(states))

    pred_seq = hmm_homemade.viterbi()[0]

    for i in range(len(pred_seq)):
        if pred_seq[i] == true_seq_activity[i]:
            d[pred_seq[i]] = [(d[pred_seq[i]][0] + 1) / d[pred_seq[i]][1], d[pred_seq[i]][1]]

    print(d)
    print(sum(pred_seq == true_seq_activity) / len(true_seq_activity))

'''
d = dict_creator(np.asarray(true_seq_activity), np.asarray(states))
pred_seq_1 = hmm_homemade.viterbi()[0]
for i in range(len(pred_seq_1)):
    if pred_seq_1[i] == true_seq_activity[i]:
        d[pred_seq_1[i]] = [(d[pred_seq_1[i]][0] + 1) / d[pred_seq_1[i]][1], d[pred_seq_1[i]][1]]


print(hmm_homemade.viterbi())
hmm_homemade.plot(hmm_homemade.get_likelihood())
'''


# FIRST STAGE
# hidden_states ==> Location
# observation ==> position of each sensor on
# states = unique_observation(location_obs)

# obs_first = location_obs
# emit = unique_observation(obs_first)
# print(states)

# start_prob, trans_prob, emit_prob = hyper_parameters(obs_first, states, emit)
# first_stage = HmmBuilder(obs_first, states, trans_prob, start_prob, emit_prob)

# pred_seq = first_stage.viterbi()[0]
# print(pred_seq)
# print(sum(pred_seq != location_obs) / len(location_obs))

# SECOND STAGE
# obs_second = sensors_obs
# emit = unique_observation(obs_second)
# for locs in pred_seq:

#    act_group = df.Action.loc[df.Location == locs].unique()

    # start_prob, trans_prob, emit_prob = hyper_parameters(obs_first, act_group, emit)

    # second_stage = HmmBuilder(obs_second, states, trans_prob, start_prob, emit_prob)
#    break
# print(act_group)
# print(true_seq_activity)
# states_first = unique_observation(df.Action.loc[df.Location == 'Bathroom'])

# observations_two = np.asarray(data[4][data[1] == days[0]])
# true_seq_state = np.asarray(data[0][data[1] == days[0]])
# states = unique_observation(true_seq_state)
# print(observations, len(observations), states, len(states))

# pi = np.random.dirichlet(np.ones(len(states)))

# A = np.random.dirichlet(np.ones(len(states)), size=len(states))

# emit = unique_observation(observations)
# B = np.random.dirichlet(np.ones(len(emit)), size=len(states))
# new_start_prob_1, new_trans_prob_1, new_emis_prob_1, likelihoods_1 = hmm_numpy(states, observations, pi, A, B)

# prova = HmmBuilder(observations, states, A, pi, B)
# print(prova.viterbi())

# print('true', true_seq_state)
# print(prova.plot(prova.get_likelihood()))

# pred_seq = prova.viterbi()[0]

# print(sum(pred_seq != true_seq_state) / len(true_seq_state))

# pred, prob = viterbi(observations, states, new_start_prob_1, new_trans_prob_1, new_emis_prob_1)
# print(likelihoods_1)

# print(new_start_prob_1)
# plt.plot(likelihoods_1)
# plt.show()
'''
# INSERISCO STANZA IN CUI SI TROVA IL SENSORE (Lx)
sens_code = s['cod_sensor']
loc = s['location']
data['locs'] = np.asarray(data[4])
for code in range(len(sens_code)):
    to_change = data[4] == sens_code[code]
    for item in to_change:
        if item:
            data['locs'][data[4] == sens_code[code]] = loc[code]

# INSERISCO CODICE DEL GRUPPO DI AZIONI  (G)
action = activity_list['Subcategory']
G = activity_list['Category']
data['groups'] = data[0]
for act in range(len(action)):
    to_change = data[0] == action[act]
    for item in to_change:
        if item:
            data['groups'][data[0] == action[act]] = G[act]

# SEQUENCE TO TRY
day_1 = data.loc[data[1] == data[1][0]]
G = day_1['groups'].unique()
cat = {}
for i in range(len(day_1['locs'].unique())):
    cat[day_1['locs'].unique()[i]] = i

for i in range(len(day_1['locs'])):
    day_1['locs'][i] = cat[day_1['locs'][i]]

print(day_1)'''
