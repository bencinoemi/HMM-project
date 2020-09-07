import pandas as pd
from datetime import datetime
from hmm_class import *
from sklearn.metrics import confusion_matrix

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
date = [''] * len(data_2.Date)
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
    for el in seq:
        if el not in unique_seq:
            unique_seq.append(el)
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
        if den == 0:
            den += 1
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
        den = sum(np.asarray(true_seq_activity) == states[s])
        if den == 0:
            den += 1
        b[s] = b[s] / den
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
        prev_seq = hmm_homemade.viterbi_to_test(obs_to_test, states, start_p, trans_p, emit_p)[0]
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


def training_hmm_three(train, train_dates, start_p_finale, trans_p_finale, emit_p_finale, all_states, all_emits):
    count_good = 0
    activity_seq = list(train.Action.values)
    states = unique_observation(activity_seq)
    n_train = train.shape[0]
    den = 0
    final_dict = {}

    for day in range(len(train_dates)):
        daily_obs = np.asarray(list(train.loc[train.Date == train_dates[day]].Sensors_code.values))
        daily_activity_seq = list(train.loc[train.Date == train_dates[day]].Action.values)
        daily_states = unique_observation(daily_activity_seq)

        daily_emits = unique_observation(np.asarray(daily_obs))

        daily_start_p = init_start_prob(all_states, daily_activity_seq) + 1e-12
        daily_trans_p = init_trans_prob(daily_activity_seq, all_states) + 1e-12
        daily_emit_p = init_emit_prob(all_emits, all_states, daily_obs, daily_activity_seq) + 1e-12

        hmm_homemade = HmmBuilder(daily_obs, all_states, daily_start_p, daily_trans_p, daily_emit_p, all_emits)

        start_p_finale = final_start_p(daily_states, start_p_finale, hmm_homemade.get_start_prob(), n_train)
        trans_p_finale = final_trans_p(daily_states, trans_p_finale, hmm_homemade.get_trans_prob(), n_train)
        emit_p_finale = final_emit_p(daily_states, emit_p_finale, hmm_homemade.get_emis_prob(), daily_emits, n_train)

        dict_good = dict_creator(np.asarray(daily_activity_seq), np.asarray(all_states))

        prev_seq = hmm_homemade.viterbi()[0]
        for act in range(len(prev_seq)):
            den += 1
            action_prev = prev_seq[act]
            if action_prev == daily_activity_seq[act]:
                dict_good[action_prev][0] = dict_good[action_prev][0] + 1
                dict_good[action_prev][2] = dict_good[action_prev][0] / dict_good[action_prev][1]
                count_good += 1

        for key in dict_good.keys():
            if key not in final_dict.keys():
                final_dict[key] = [0, 0, 0.]
            final_dict[key][0] = final_dict[key][0] + dict_good[key][0]
            final_dict[key][1] = final_dict[key][1] + dict_good[key][1]
            print(final_dict[key])
            if final_dict[key][1] != 0:
                final_dict[key][2] = final_dict[key][0] / final_dict[key][1]

        # print(dict_good)
    print('final_dict', final_dict)
    print('train_accuracy', count_good / den)
    return start_p_finale, trans_p_finale, emit_p_finale, count_good / den

def test_hmm_three(test, start_p_finale, trans_p_finale, emit_p_finale, all_states, all_emits):
    count_test, den = 0, 0
    test_obs = list(test.Sensors_code.values)
    true_test = list(test.Action.values)
    states_test = unique_observation(true_test)

    hmm = HmmBuilder(obs=test_obs, states=all_states, start_probability=start_p_finale,
                     transition_probability=trans_p_finale, emission_probability=emit_p_finale, emit=all_emits)

    prev_test = hmm.viterbi_to_test(test_obs, all_states, start_p_finale, trans_p_finale, emit_p_finale)[0]

    dict_good = dict_creator(np.asarray(true_test), np.asarray(all_states))

    for act in range(len(prev_test)):
        den += 1
        prev_action = prev_test[act]
        if prev_action == true_test[act]:
            dict_good[prev_action][0] = dict_good[prev_action][0] + 1

            dict_good[prev_action][2] = dict_good[prev_action][0] / dict_good[prev_action][1]
            count_test += 1

    print(dict_good)
    print('test_accuracy', count_test / den)
    return prev_test, count_test / den

def training_hmm_four(train, test, train_dates, start_p_finale, trans_p_finale, emit_p_finale, all_states, all_emits):
    count_good = 0
    activity_seq = list(train.Action.values)
    states = unique_observation(activity_seq)
    n_train = train.shape[0]
    den = 0
    final_dict = {}
    max_prob = -1
    for day in range(len(train_dates)):

        daily_obs = np.asarray(list(train.loc[train.Date == train_dates[day]].Sensors_code.values))
        daily_activity_seq = list(train.loc[train.Date == train_dates[day]].Action.values)
        daily_states = unique_observation(daily_activity_seq)

        daily_emits = unique_observation(np.asarray(daily_obs))

        daily_start_p = init_start_prob(all_states, daily_activity_seq) + 1e-12
        daily_trans_p = init_trans_prob(daily_activity_seq, all_states) + 1e-12
        daily_emit_p = init_emit_prob(all_emits, all_states, daily_obs, daily_activity_seq) + 1e-12

        hmm_homemade = HmmBuilder(daily_obs, all_states, daily_start_p, daily_trans_p, daily_emit_p, all_emits)

        start_p_finale = hmm_homemade.get_start_prob()
        trans_p_finale = hmm_homemade.get_trans_prob()
        emit_p_finale = hmm_homemade.get_emis_prob()

# TODO: trova il modo di assegnare le top se non dovesse mai essere migliore

        prev_seq_prob = first_test_hmm(test, start_p_finale, trans_p_finale, emit_p_finale, all_states, all_emits)

        if prev_seq_prob > max_prob:
            max_prob = prev_seq_prob
            top_start_p = start_p_finale
            top_trans_p = trans_p_finale
            top_emit_p = emit_p_finale

        dict_good = dict_creator(np.asarray(daily_activity_seq), np.asarray(all_states))

        prev_seq = hmm_homemade.viterbi()[0]
        for act in range(len(prev_seq)):
            den += 1
            action_prev = prev_seq[act]
            if action_prev == daily_activity_seq[act]:
                dict_good[action_prev][0] = dict_good[action_prev][0] + 1
                dict_good[action_prev][2] = dict_good[action_prev][0] / dict_good[action_prev][1]
                count_good += 1

        for key in dict_good.keys():
            if key not in final_dict.keys():
                final_dict[key] = [0, 0, 0.]
            final_dict[key][0] = final_dict[key][0] + dict_good[key][0]
            final_dict[key][1] = final_dict[key][1] + dict_good[key][1]
            # print(final_dict[key])
            if final_dict[key][1] != 0:
                final_dict[key][2] = final_dict[key][0] / final_dict[key][1]

        # print(dict_good)
    print('final_dict', final_dict)
    print('train_accuracy', count_good / den)
    return top_start_p, top_trans_p, top_emit_p, count_good / den

def first_test_hmm(test, start_p_finale, trans_p_finale, emit_p_finale, all_states, all_emits):
    test_obs = list(test.Sensors_code.values)
    hmm = HmmBuilder(obs=test_obs, states=all_states, start_probability=start_p_finale,
                     transition_probability=trans_p_finale, emission_probability=emit_p_finale, emit=all_emits)

    prev_seq_prob = hmm.viterbi_to_test(test_obs, all_states, start_p_finale, trans_p_finale, emit_p_finale)[1]
    # prev_seq_prob = np.exp(hmm.get_likelihood())
    return prev_seq_prob


def test_hmm_four(test, start_p_finale, trans_p_finale, emit_p_finale, all_states, all_emits):
    count_test, den = 0, 0
    test_obs = list(test.Sensors_code.values)
    true_test = list(test.Action.values)
    states_test = unique_observation(true_test)

    hmm = HmmBuilder(obs=test_obs, states=all_states, start_probability=start_p_finale,
                     transition_probability=trans_p_finale, emission_probability=emit_p_finale, emit=all_emits)

    prev_test = hmm.viterbi_to_test(test_obs, all_states, start_p_finale, trans_p_finale, emit_p_finale)[0]

    dict_good = dict_creator(np.asarray(true_test), np.asarray(all_states))

    for act in range(len(prev_test)):
        den += 1
        prev_action = prev_test[act]
        if prev_action == true_test[act]:
            dict_good[prev_action][0] = dict_good[prev_action][0] + 1

            dict_good[prev_action][2] = dict_good[prev_action][0] / dict_good[prev_action][1]
            count_test += 1

    print(dict_good)
    print('test_accuracy', count_test / den)
    return prev_test, count_test / den


def all_states_dict(data):
    d = {}
    for act in data.Action.unique():
        d[act] = 0.
    return d

def all_emit_dict(data):
    d = {}
    for el in data.Sensors_code.unique():
        d[el] = 0.
    return d

def init_start_p_last(data):
    return all_states_dict(data)

def init_trans_p_last(data):
    d = all_states_dict(data)
    for st in d.keys():
        d[st] = all_states_dict(data)
    return d

def init_emit_p_last(data):
    d = all_states_dict(data)
    for st in d.keys():
        d[st] = all_emit_dict(data)
    return d

def final_start_p(states, d, start_p, n_train):
    for st in range(len(states)):
        d[states[st]] += start_p[st] / n_train
    return d

def final_trans_p(states, d, trans_p, n_train):
    for st in range(len(states)):
        for s in range(len(states)):
            d[states[st]][states[s]] = trans_p[st, s] / n_train
    return d

def final_emit_p(states, d, emit_p, emits, n_train):
    for st in range(len(states)):
        for em in range(len(emits)):
            d[states[st]][emits[em]] = emit_p[st, em] / n_train
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
    idx_st = 0
    for st in trans_p.keys():
        idx_s = 0
        for s in trans_p[st].keys():
            trans_p_array[idx_st, idx_s] = trans_p[st][s]
            idx_s += 1
        idx_st += 1

    return trans_p_array

def emit_p_converter(emit_p: dict, data):
    emit_p_array = np.zeros((len(emit_p), len(all_emit_dict(data))))
    idx_st = 0
    for st in emit_p.keys():
        idx_em = 0
        for em in emit_p[st].keys():
            emit_p_array[idx_st, idx_em] = emit_p[st][em]
            idx_em += 1
        idx_st += 1
    return emit_p_array

# true_seq_activity = list(data_1.loc[data_1.Date == dates_1[1]].Action.values)
# states = unique_observation(true_seq_activity)
# start_p = init_start_prob(states, true_seq_activity) + 1e-12

def confusion_matrix_by_me(conf_dict: dict, prev_seq, true_seq):
    observed_acts = {}
    for t in range(len(prev_seq)):
        true_activity = true_seq[t]
        prev_activity = prev_seq[t]
        den = sum(true_seq == true_activity)
        if den == 0:
            den += 1
        conf_dict[true_activity][prev_activity] += 1 / den

        if true_activity not in observed_acts:
            observed_acts[true_activity] = 0
        observed_acts[true_activity] += 1

    return conf_dict, observed_acts

def global_training(data):
    dates = data.Date.unique()
    true_all_seq = data.Action.values
    all_states = unique_observation(true_all_seq)
    all_emits = unique_observation(data.Sensors_code.values)
    test_acc = []
    train_acc = []
    conf_dict = init_trans_p_last(data)
    final_prev_seq = np.array([])
    final_true_seq = np.array([])
    for day in range(len(dates)):
        train_data = data.loc[data.Date != dates[day]]
        test_data = data.loc[data.Date == dates[day]]
        true_seq = test_data.Action.values
        start_p_finale = init_start_p_last(data)
        trans_p_finale = init_trans_p_last(data)
        emit_p_finale = init_emit_p_last(data)

        start_p_finale, trans_p_finale, emit_p_finale, daily_train_acc = training_hmm_three(train_data,
                                                                                            train_data.Date.unique(),
                                                                                            start_p_finale,
                                                                                            trans_p_finale,
                                                                                            emit_p_finale,
                                                                                            all_states, all_emits)

        train_acc.append(daily_train_acc)
        start_p_finale = start_p_converter(start_p_finale)
        trans_p_finale = trans_p_converter(trans_p_finale)
        emit_p_finale = emit_p_converter(emit_p_finale, data)
        prev_seq, acc = test_hmm_three(test_data, start_p_finale, trans_p_finale, emit_p_finale, all_states, all_emits)
        test_acc.append(acc)
        final_prev_seq = np.append(final_prev_seq, prev_seq)
        final_true_seq = np.append(final_true_seq, true_seq)
        # conf_dict = confusion_matrix(conf_dict, prev_seq, true_seq)
    # print(conf_dict)
    print(confusion_matrix_by_me(conf_dict,  final_prev_seq, final_true_seq))
    return test_acc, train_acc


def global_training_four(data):
    dates = data.Date.unique()
    true_all_seq = data.Action.values
    all_states = unique_observation(true_all_seq)
    all_emits = unique_observation(data.Sensors_code.values)
    test_acc = []
    train_acc = []
    final_prev_seq = np.array([])
    final_true_seq = np.array([])
    conf_dict = init_trans_p_last(data)

    for day in range(len(dates)):
        train_data = data.loc[data.Date != dates[day]]
        test_data = data.loc[data.Date == dates[day]]
        true_seq = test_data.Action.values
        start_p_finale = init_start_p_last(data)
        trans_p_finale = init_trans_p_last(data)
        emit_p_finale = init_emit_p_last(data)

        start_p_finale, trans_p_finale, emit_p_finale, daily_train_acc = training_hmm_four(train_data, test_data,
                                                                                           train_data.Date.unique(),
                                                                                           start_p_finale,
                                                                                           trans_p_finale,
                                                                                           emit_p_finale,
                                                                                           all_states, all_emits)

        train_acc.append(daily_train_acc)

        prev_seq, acc = test_hmm_four(test_data, start_p_finale, trans_p_finale, emit_p_finale, all_states, all_emits)
        test_acc.append(acc)

        final_prev_seq = np.append(final_prev_seq, prev_seq)
        final_true_seq = np.append(final_true_seq, true_seq)

    conf_matr, observ_acts = confusion_matrix_by_me(conf_dict, final_prev_seq, final_true_seq)
    matr = np.zeros((len(all_states), len(all_states)))
    arr = np.zeros(len(all_states))
    i = 0
    for act in conf_matr.keys():
        j = 0
        for acc in conf_matr[act].keys():
            matr[i, j] = conf_matr[act][acc]
            j += 1
        arr[i] = observ_acts[act]
        i += 1
    print(matr)
    print(arr)
    return test_acc, train_acc


print(global_training_four(data_2))

# Train_acc(data_1):
# [0.874813153961136, 0.8786936236391913, 0.8734802431610942, 0.8786722624952308, 0.8702556158017041,
# 0.872613946240748, 0.878168747635263, 0.8708223807735637, 0.8723404255319149, 0.8743559254855331,
# 0.8707403055229143, 0.8739715781600599, 0.8750937734433608, 0.8663755458515284, 0.8704407294832827, 0.87782302850796]
# Test_acc (data_1):
# [0.3333333333333333, 0.06, 0.11428571428571428, 0.0728476821192053, 0.0, 0.25853658536585367,
# 0.21705426356589147, 0.1559633027522936, 0.18716577540106952, 0.11646586345381527, 0.0639269406392694,
# 0.04081632653061224, 0.05660377358490566, 0.07053941908713693, 0.06428571428571428, 0.16901408450704225]
# TEST ACC 05/09:
# [0.09375, 0.04, 0.07857142857142857, 0.2052980132450331, 0.02631578947368421, 0.2634146341463415,
# 0.03875968992248062, 0.01834862385321101, 0.1443850267379679, 0.0963855421686747, 0.091324200913242,
# 0.12244897959183673, 0.018867924528301886, 0.1078838174273859, 0.05, 0.0], (6 better than before)


# Train_acc (data_2):
# [0.8578378378378378, 0.8611838658983761, 0.8635875402792696, 0.8620689655172413, 0.8607594936708861,
# 0.8597826086956522, 0.8657357679914071, 0.8718766613503456, 0.8761415525114156, 0.8699453551912568,
# 0.871038251366120, 0.8567596566523605, 0.8426365795724465, 0.861286919831223, 0.8635650810245687, 0.8775623268698061]
# Test_acc (data_2):
# [0.5357142857142857, 0.07547169811320754, 0.01, 0.10416666666666667, 0.0625, 0.1721311475409836,
# 0.02, 0.32098765432098764, 0.07142857142857142, 0.0, 0.0, 0.11224489795918367, 0.0683453237410072,
# 0.045454545454545456, 0.20408163265306123, 0.06369426751592357]
# TEST ACC 05/09:
# [0.008928571428571428, 0.03773584905660377, 0.11, 0.0, 0.03125, 0.00819672131147541, 0.02, 0.0,
# 0.004761904761904762, 0.05303030303030303, 0.12878787878787878, 0.05102040816326531, 0.03597122302158273,
# 0.0, 0.02040816326530612, 0.0] (three better than before)

# Final test_acc 05/09:
# [0.4642857142857143, 0.16981132075471697, 0.03, 0.0, 0.12946428571428573, 0.08196721311475409, 0.14,
# 0.024691358024691357, 0.02857142857142857, 0.0, 0.11363636363636363, 0.09183673469387756, 0.08273381294964029,
# 0.07575757575757576, 0.3469387755102041, 0.3375796178343949]






