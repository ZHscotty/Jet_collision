import numpy as np
import pandas as pd
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_data(type):
    data_dir = './data'
    if type == 'train':
        data = pd.read_csv(os.path.join(data_dir, 'simple_train_R04_jet.csv'))
        label = data['label'].values
        label2index = {1: 0, 4: 1, 5: 2, 21: 3}
        labelSet = [label2index[x] for x in label]
        labelSet = to_categorical(labelSet)
    else:
        data = pd.read_csv(os.path.join(data_dir, 'simple_test_R04_jet.csv'))
        id = data['jet_id'].values

    dataSet = data[['number_of_particles_in_this_jet', 'jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass']].values

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



