import numpy as np
import pandas as pd
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data(type):
    data_dir = 'E:\\比赛数据\\jet_simple_data'
    if type == 'train':
        data = pd.read_csv(os.path.join(data_dir, 'simple_train_R04_jet.csv'))
        label = data['label'].values
        label2index = {1: 0, 4: 1, 5: 2, 21: 3}
        labelSet = [label2index[x] for x in label]
        labelSet = to_categorical(labelSet)
    else:
        data = pd.read_csv(os.path.join(data_dir, 'simple_test_R04_jet.csv'))
        id = data['jet_id'].values

    data_new = feature_process(data)
    dataSet = data_new[['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass',
                        'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']].values

    if type == 'train':
        x_train, x_test, y_train, y_test = train_test_split(dataSet, labelSet, test_size=0.2, random_state=42,
                                                            stratify=labelSet)
        return x_train, x_test, y_train, y_test

    else:
        return dataSet, id


def output_data(predict, id):
    index2label = {0: 1, 1: 4, 2: 5, 3: 21}
    predict = [index2label[x] for x in predict]
    output = np.array([id, predict]).transpose()
    print('output shape', output.shape)
    d = pd.DataFrame(output, columns=['id', 'label'])
    d.to_csv('result.csv', encoding='utf-8', index=False, sep=',')


def feature_process(data):
    """
    :param data: DataFrame
    :return:
    """
    number = data['number_of_particles_in_this_jet'].values
    energy = data['jet_energy'].values
    mass = data['jet_mass'].values
    x = data['jet_px'].values
    y = data['jet_py'].values
    z = data['jet_pz'].values

    num_max = np.max(number)
    energy_max = np.max(energy)
    mass_max = np.max(mass)

    feature1 = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    feature2 = energy/(mass+1)
    feature3 = energy/(number+1)
    feature4 = mass/number
    feature5 = energy*energy
    feature6 = mass*mass

    data['number_of_particles_in_this_jet'] = number/num_max
    data['jet_energy'] = energy/energy_max
    data['jet_mass'] = mass/mass_max
    data['jet_px'] = x / feature1
    data['jet_py'] = y / feature1
    data['jet_pz'] = z / feature1

    data['feature1'] = feature1
    data['feature2'] = feature2
    data['feature3'] = feature3
    data['feature4'] = feature4
    data['feature5'] = feature5
    data['feature6'] = feature6
    return data


def data_analize(data):
    energy = data['jet_energy'].values
    mass = data['jet_mass'].values
    label = data['label'].values
    c = []
    for x in label:
        if x == 1:
            c.append('r')
        elif x == 4:
            c.append('g')
        elif x == 5:
            c.append('b')
        elif x == 21:
            c.append('y')
        else:
            print(type(x))
            print('error')
            break
    plt.scatter(energy, mass, c=c, linewidths=0.1)
    plt.savefig('./pic/analize/energy-mass.png')
    plt.show()


if __name__ == '__main__':
    data_dir = 'E:\\比赛数据\\jet_simple_data'
    data = pd.read_csv(os.path.join(data_dir, 'simple_train_R04_jet.csv'))
    data = feature_process(data)
    pd.set_option('display.max_columns', None)
    print(data)
