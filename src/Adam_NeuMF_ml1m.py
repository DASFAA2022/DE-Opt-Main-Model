import argparse
import tensorflow.compat.v1 as tf

import sys
import os
# import os.path
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



from model.Adam_NeuMF import *
from load_data_ranking import *


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--model', choices=['NeuMF', 'LRML', 'DE_NeuMF', 'DE_LRML'],
                        default='Adam_NeuMF')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_factors', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=512)  # 128 for unlimpair
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 1e-4 for unlimpair
    parser.add_argument('--reg_rate', type=float, default=0.1)  # 0.01 for unlimpair
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    num_factors = args.num_factors
    display_step = args.display_step
    batch_size = args.batch_size

    train_data, validation_data, test_data, n_user, n_item = load_data_neg(path='/home/sunbo/DE_Framwork/data/ml1m/ratings.dat', test_size=0.2, sep="::")


    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True

    for i in [0.1, 0.01, 0.001, 0.0001]:
        for j in [10, 1]:
            learning_rate = i
            reg_rate = j

            with tf.Session(config=config) as sess:
                model = None
                if args.model == "Adam_NeuMF":
                    model = Adam_NeuMF(sess, n_user, n_item, learning_rate, reg_rate)
                    model.build_network()
                    model.execute(train_data, validation_data, test_data)

        # build and execute the model
        # if model is not None:
        #     model.build_network()
        #     model.execute(train_data, validation_data, test_data)
