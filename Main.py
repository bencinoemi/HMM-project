import pandas as pd
from datetime import datetime
from hmm_class import *

sensors = pd.read_csv('/home/noe/Università/in_corso/Machine Learning/progetto/dataset/subject1/sensors.csv',
                      header=None, names=['Sensors_code', 'Location', 'Sensors_name'])

activity_list = pd.read_csv('/home/noe/Università/in_corso/Machine Learning/progetto/dataset/subject1/Activities.csv')

data = pd.read_csv('/home/noe/Università/in_corso/Machine Learning/progetto/dataset/subject1/activity_sub1.csv',
                   header=None, names=['Action', 'Date', 'Activity_start', 'Activity_end', 'Sensors_code',
                                       'Sensors_name', 'start_time', 'end_time'])

# print(s)
# print(activity_list)
# print(data.head())

activities = np.asarray(activity_list['Subcategory'])
# print(activities)

# data[1] = pd.to_datetime(data[1]).dt.date
# print(data[1].unique())
# days = data[1].unique()

data['Date'] = [datetime.strptime(x, "%m/%d/%Y").date() for x in data['Date']]
data['start_time'] = [datetime.strptime(x, "%H:%M:%S").time() for x in data['start_time']]
data['end_time'] = [datetime.strptime(x, "%H:%M:%S").time() for x in data['end_time']]


df = pd.merge(data, sensors)
df = df.sort_values(['Date', 'start_time'])

dates = df.Date.unique()
times = df.start_time.unique()

def unique_observation(obs):
    emis = []
    for i in obs:
        if i not in emis:
            emis.append(i)
    return np.asarray(emis)

def hyper_parameters(obs, hidden_states, emit):
    pi = np.random.dirichlet(np.ones(len(hidden_states)))

    A = np.random.dirichlet(np.ones(len(hidden_states)), size=len(hidden_states))

    B = np.random.dirichlet(np.ones(len(emit)), size=len(hidden_states))
    return pi, A, B


for day in range(len(dates)):
    sensors_obs = np.asarray(df.loc[df.Date == dates[day]].Sensors_code)
    true_seq_activity = np.asarray(df.loc[df.Date == dates[day]].Action)

    states = unique_observation(true_seq_activity)
    observations = sensors_obs
    emits = unique_observation(observations)


# print(len(states))

start_p, trans_p, emit_p = hyper_parameters(observations, states, emits)

hmm_homemade = HmmBuilder(observations, states, trans_p, start_p, emit_p)
pred_seq = hmm_homemade.viterbi()[0]
print(sum(pred_seq == true_seq_activity) / len(true_seq_activity))


'''
states = unique_observation(true_seq_activity)
observations = sensors_obs
emits = unique_observation(observations)

start_p, trans_p, emit_p = hyper_parameters(observations, states, emits)

hmm_homemade = HmmBuilder(observations, states, trans_p, start_p, emit_p)
print(hmm_homemade.viterbi())
pred_seq = hmm_homemade.viterbi()[0]
# print(pred_seq)
# print(true_seq_activity)
print(sum(pred_seq == true_seq_activity) / len(true_seq_activity))
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
