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

data_1['Date'] = pd.to_datetime(data_1['Date'], format='%m/%d/%Y')
data_1['start_time'] = pd.to_datetime(data_1['start_time'], format="%H:%M:%S")
data_1['end_time'] = pd.to_datetime(data_1['end_time'], format="%H:%M:%S")

df_1 = pd.merge(data_1, sensors)
df_1 = df_1.sort_values(['Date', 'start_time'])

dates_1 = df_1.Date.unique()
times_1 = df_1.start_time.unique()

data_2['Date'].unique()

# 31 of April does not exist: changed with 30 of April
date = ['']*len(data_2.Date)
for i in range(len(data_2.Date)):
    if data_2.Date[i] == '4/31/2003':
        date[i] = '4/30/2003'
    else:
        date[i] = data_2.Date[i]
data_2['Date_mod'] = date

data_2['Date'] = pd.to_datetime(data_2['Date_mod'], format='%m/%d/%Y')
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
        dct[s] = [0, sum(np.asarray(true_seq) == s), 0]
    return dct


def init_start_prob(unique_states, seq):
    # Every prior prob i is:
    # #of state i observed / #obs

    pi = []
    for s in unique_states:
        pi.append(sum(np.asarray(seq) == s))
    return np.asarray(pi) / len(seq)

def init_trans_prob(seq, unique_states):
    # Every trans prob i, j is:
    # #number of times the state i is followed by state j / #of times the state i appears

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
    # Every emit prob is:
    # #number of times a specific sensor is used in an action / # number of sensors used in that action
    b = np.zeros((len(states), len(emits)))
    for s in range(len(states)):
        for t in range(len(obs)):
            if true_seq_activity[t] == states[s]:
                for e in range(len(emits)):
                    if obs[t] == emits[e]:
                        b[s, e] += 1
        b[s] = b[s] / sum(np.asarray(true_seq_activity) == states[s])

    return b

# first try
def training_hmm(data):
    dates = data.Date.unique()
    count_good = 0
    final_dict = {}
    for day in range(len(dates)):
        obs = list(data.loc[data.Date == dates[day]].Sensors_code.values)
        true_seq_activity = list(data.loc[data.Date == dates[day]].Action.values)
        states = unique_observation(true_seq_activity)

        emits = unique_observation(np.asarray(obs))
        start_p = init_start_prob(states, true_seq_activity) + 1e-12
        trans_p = init_trans_prob(true_seq_activity, states) + 1e-12
        emit_p = init_emit_prob(emits, states, obs, true_seq_activity) + 1e-12

        hmm_homemade = HmmBuilder(np.asarray(obs), states, start_p, trans_p, emit_p)
        # hmm_homemade.plot_prob()
        dict_good = dict_creator(np.asarray(true_seq_activity), np.asarray(states))

        prev_seq = hmm_homemade.viterbi()[0]

        for act in range(len(prev_seq)):
            if prev_seq[act] == true_seq_activity[act]:
                dict_good[prev_seq[act]] = [(dict_good[prev_seq[act]][0] + 1), dict_good[prev_seq[act]][1], 0]

                dict_good[prev_seq[act]][2] = dict_good[prev_seq[act]][0] / dict_good[prev_seq[act]][1]

                if prev_seq[act] not in final_dict.keys():
                    final_dict[prev_seq[act]] = [0, 0, 0.]

                final_dict[prev_seq[act]] = [final_dict[prev_seq[act]][0] + dict_good[prev_seq[act]][0],
                                             final_dict[prev_seq[act]][1] + dict_good[prev_seq[act]][1], 0]
                final_dict[prev_seq[act]][2] = final_dict[prev_seq[act]][0] / final_dict[prev_seq[act]][1]
                count_good += 1

        print(dict_good)
    print(final_dict)
    print(count_good / data_1.shape[0])


# print('subject_1')
# training_hmm(data_1)
# print('subject_2')
# training_hmm(data_2)

# second try

def training_hmm_two(data):
    final_dict = {}
    count_good = 0
    dates = data.Date.unique()
    for d in range(len(dates)):
        obs_to_train = []
        true_seq_train = []
        obs_to_test = list(data.loc[data.Date == dates[d]].Sensors_code.values)
        true_seq_test = list(data.loc[data.Date == dates[d]].Action.values)
        for day in range(len(dates)):
            if dates[day] != dates[d]:
                obs_to_train.extend(list(data.loc[data.Date == dates[day]].Sensors_code.values))
                true_seq_train.extend(list(data.loc[data.Date == dates[day]].Action.values))

        states = unique_observation(true_seq_train)
        emits = unique_observation(np.asarray(obs_to_train))
        start_p = init_start_prob(states, true_seq_train) + 1e-12
        trans_p = init_trans_prob(true_seq_train, states) + 1e-12
        emit_p = init_emit_prob(emits, states, obs_to_train, true_seq_train) + 1e-12

        hmm_homemade = HmmBuilder(np.asarray(obs_to_train), states, start_p, trans_p, emit_p)
        prev_seq = hmm_homemade.viterbi_to_test(obs_to_test)[0]
        # print('predicted', prev_seq)
        # print('true', true_seq_test)

        dict_good = dict_creator(np.asarray(true_seq_test), np.asarray(states))

        for act in range(len(prev_seq)):
            if prev_seq[act] == true_seq_test[act]:
                dict_good[prev_seq[act]] = [(dict_good[prev_seq[act]][0] + 1), dict_good[prev_seq[act]][1], 0]

                dict_good[prev_seq[act]][2] = dict_good[prev_seq[act]][0] / dict_good[prev_seq[act]][1]

                if prev_seq[act] not in final_dict.keys():
                    final_dict[prev_seq[act]] = [0, 0, 0.]

                final_dict[prev_seq[act]] = [final_dict[prev_seq[act]][0] + dict_good[prev_seq[act]][0],
                                             final_dict[prev_seq[act]][1] + dict_good[prev_seq[act]][1], 0]
                final_dict[prev_seq[act]][2] = final_dict[prev_seq[act]][0] / final_dict[prev_seq[act]][1]
                count_good += 1

        print(final_dict)
    print(count_good / data_1.shape[0])


# print('subject_1')
# training_hmm_two(data_1) # 0.10
# print('subject_2')
# training_hmm_two(data_2)  # 0.048


def training_hmm_three(train, train_dates, start_p_finale, trans_p_finale, emit_p_finale):
    count_good = 0
    final_dict = {}
    n_train = train.shape[0]

    for day in range(len(train_dates)):
        daily_obs = list(train.loc[train.Date == train_dates[day]].Sensors_code.values)
        daily_activity_seq = list(train.loc[train.Date == train_dates[day]].Action.values)
        daily_states = unique_observation(daily_activity_seq)

        daily_emits = unique_observation(np.asarray(daily_obs))
        daily_start_p = init_start_prob(daily_states, daily_activity_seq) + 1e-12
        daily_trans_p = init_trans_prob(daily_activity_seq, daily_states) + 1e-12
        daily_emit_p = init_emit_prob(daily_emits, daily_states, daily_obs, daily_activity_seq) + 1e-12

        hmm_homemade = HmmBuilder(np.asarray(daily_obs), daily_states, daily_start_p, daily_trans_p, daily_emit_p)

        start_p_finale = final_start_p(daily_states, start_p_finale, hmm_homemade.get_start_prob(), n_train)
        trans_p_finale = final_trans_p(daily_states, trans_p_finale, hmm_homemade.get_trans_prob(), n_train)
        emit_p_finale = final_emit_p(daily_states, emit_p_finale, hmm_homemade.get_emis_prob(), daily_emits, n_train)

        dict_good = dict_creator(np.asarray(daily_activity_seq), np.asarray(daily_states))

        prev_seq = hmm_homemade.viterbi()[0]
        for act in range(len(prev_seq)):
            if prev_seq[act] == daily_activity_seq[act]:
                dict_good[prev_seq[act]] = [(dict_good[prev_seq[act]][0] + 1), dict_good[prev_seq[act]][1], 0]

                dict_good[prev_seq[act]][2] = dict_good[prev_seq[act]][0] / dict_good[prev_seq[act]][1]

                if prev_seq[act] not in final_dict.keys():
                    final_dict[prev_seq[act]] = [0, 0, 0.]

                final_dict[prev_seq[act]] = [final_dict[prev_seq[act]][0] + dict_good[prev_seq[act]][0],
                                             final_dict[prev_seq[act]][1] + dict_good[prev_seq[act]][1], 0]
                final_dict[prev_seq[act]][2] = final_dict[prev_seq[act]][0] / final_dict[prev_seq[act]][1]
                count_good += 1

        # print(dict_good)
    print('final_dict', final_dict)
    print('train_accuracy', count_good / n_train)
    return start_p_finale, trans_p_finale, emit_p_finale, count_good / n_train

def test_hmm_three(test, start_p_finale, trans_p_finale, emit_p_finale, all_states):
    count_test = 0
    test_obs = list(test.Sensors_code.values)
    true_test = list(test.Action.values)
    states_test = unique_observation(true_test)

    hmm = HmmBuilder(test_obs, states_test, start_p_finale, trans_p_finale, emit_p_finale)

    prev_test = hmm.viterbi_to_test(test_obs, all_states, start_p_finale, trans_p_finale, emit_p_finale)[0]

    dict_good = dict_creator(np.asarray(true_test), np.asarray(states_test))

    for act in range(len(prev_test)):
        if prev_test[act] == true_test[act]:
            dict_good[prev_test[act]] = [(dict_good[prev_test[act]][0] + 1), dict_good[prev_test[act]][1], 0]

            dict_good[prev_test[act]][2] = dict_good[prev_test[act]][0] / dict_good[prev_test[act]][1]
            count_test += 1

    print(dict_good)
    print('test_accuracy', count_test / len(prev_test))
    return count_test / len(prev_test)

def all_states_dict(data):
    d = {}
    for i in data.Action.unique():
        d[i] = 0.
    return d

def all_emis_dict(data):
    d = {}
    for i in data.Sensors_code.unique():
        d[i] = 0.
    return d

def init_start_p_last(data):
    return all_states_dict(data)

def init_trans_p_last(data):
    d = all_states_dict(data)
    for i in d.keys():
        d[i] = all_states_dict(data)
    return d

def init_emit_p_last(data):
    d = all_states_dict(data)
    for i in d.keys():
        d[i] = all_emis_dict(data)
    return d

def final_start_p(states, d, start_p, n_train):
    for i in range(len(states)):
        d[states[i]] += start_p[i] / n_train
    return d

def final_trans_p(states, d, trans_p, n_train):
    for i in range(len(states)):
        for j in range(len(states)):
            d[states[i]][states[j]] = trans_p[i, j] / n_train
    return d

def final_emit_p(states, d, emit_p, emits, n_train):
    for i in range(len(states)):
        for j in range(len(emits)):
            d[states[i]][emits[j]] = emit_p[i, j] / n_train
    return d

def start_p_converter(start_p: dict):
    start_p_array = np.zeros(len(start_p))
    index = 0
    for key in start_p.keys():
        start_p_array[index] = start_p[key]
        index += 1
    return start_p_array

def trans_p_converter(trans_p: dict):
    trans_p_array = np.zeros((len(trans_p), len(trans_p)))
    j = 0
    for i in trans_p.keys():
        k = 0
        for l in trans_p[i].keys():
            trans_p_array[j, k] = trans_p[i][l]
            k += 1
        j += 1

    return trans_p_array

def emit_p_converter(emit_p: dict, data):
    emit_p_array = np.zeros((len(emit_p), len(all_emis_dict(data))))
    j = 0
    for i in emit_p.keys():
        k = 0
        for l in emit_p[i].keys():
            emit_p_array[j, k] = emit_p[i][l]
            k += 1
        j += 1
    return emit_p_array

# true_seq_activity = list(data_1.loc[data_1.Date == dates_1[1]].Action.values)
# states = unique_observation(true_seq_activity)
# start_p = init_start_prob(states, true_seq_activity) + 1e-12


def global_training(data):
    dates = data.Date.unique()
    all_states = data.Action.unique()
    test_acc = []
    train_acc = []
    for day in range(len(dates)):
        train_data = data.loc[data.Date != dates[day]]
        test_data = data.loc[data.Date == dates[day]]
        start_p_finale = init_start_p_last(data)
        trans_p_finale = init_trans_p_last(data)
        emit_p_finale = init_emit_p_last(data)

        start_p_finale, trans_p_finale, emit_p_finale, daily_train_acc = training_hmm_three(train_data,
                                                                                            train_data.Date.unique(),
                                                                                            start_p_finale,
                                                                                            trans_p_finale,
                                                                                            emit_p_finale)
        train_acc.append(daily_train_acc)
        start_p_finale = start_p_converter(start_p_finale)
        trans_p_finale = trans_p_converter(trans_p_finale)
        emit_p_finale = emit_p_converter(emit_p_finale, data)
        test_acc.append(test_hmm_three(test_data, start_p_finale, trans_p_finale, emit_p_finale, all_states))
    return test_acc, train_acc


print(global_training(data_1))

# Test_acc (data_1):
# [0.3333333333333333, 0.06, 0.11428571428571428, 0.0728476821192053, 0.0, 0.25853658536585367,
# 0.21705426356589147, 0.1559633027522936, 0.18716577540106952, 0.11646586345381527, 0.0639269406392694,
# 0.04081632653061224, 0.05660377358490566, 0.07053941908713693, 0.06428571428571428, 0.16901408450704225]

# Test_acc (data_2):
# [0.5357142857142857, 0.07547169811320754, 0.01, 0.10416666666666667, 0.0625, 0.1721311475409836,
# 0.02, 0.32098765432098764, 0.07142857142857142, 0.0, 0.0, 0.11224489795918367, 0.0683453237410072,
# 0.045454545454545456, 0.20408163265306123, 0.06369426751592357]



