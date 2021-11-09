import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

def load_data_rating(path="../data/ml100k/movielens_100k.dat", header=['user_id', 'item_id', 'rating', 'category'],
                     test_size=0.2, sep="::"):
    '''
        Loading the data for rating prediction task
        :param path: the path of the dataset, datasets should be in the CSV format
        :param header: the header of the CSV format, the first three should be: user_id, item_id, rating
        :param test_size: the test ratio, default 0.1
        :param sep: the seperator for csv colunms, defalut space
        :return:
        '''

    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]


    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data, validation_data = train_test_split(train_data, test_size=0.125)
    train_data = pd.DataFrame(train_data)
    validation_data = pd.DataFrame(validation_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []
    # n_users = 0
    # n_items = 0

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        # n_users = max(n_users, u) + 1
        # n_items = max(n_items, i) + 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])
    print(n_users, n_items)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    validation_row = []
    validation_col = []
    validation_rating = []

    for line in validation_data.itertuples():
        validation_row.append(line[1] - 1)
        validation_col.append(line[2] - 1)
        validation_rating.append(line[3])
    validation_matrix = csr_matrix((validation_rating, (validation_row, validation_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), validation_matrix.todok(), test_matrix.todok(), n_users, n_items
