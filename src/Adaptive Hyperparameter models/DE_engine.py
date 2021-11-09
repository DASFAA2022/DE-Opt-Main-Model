import os
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
import torch
from factorizer.factorizer import setup_factorizer
from regularizer.regularizer import setup_regularizer
from utils.data_loader import setup_sample_generator
from utils.mp_cuda_evaluate import evaluate_ui_uj_df

import warnings

warnings.filterwarnings('ignore')

def setup_args(parser=None):
    """ Set up arguments for the Engine

    return:
        python dictionary
    """
    if parser is None:
        parser = ArgumentParser()
    data = parser.add_argument_group('Data')
    engine = parser.add_argument_group('Engine Arguments')
    factorize = parser.add_argument_group('Factorizer Arguments')
    matrix_factorize = parser.add_argument_group('MF Arguments')
    regularize = parser.add_argument_group('Regularizer Arguments')
    log = parser.add_argument_group('Tensorboard Arguments')

    engine.add_argument('--alias', default='experiment',
                        help='Name for the experiment')

    data.add_argument('--data-path', default='./data/ml-1m/ratings.dat')
    data.add_argument('--data-type', default='ml1m-mf', help='type of the dataset')
    data.add_argument('--filtered-data-path', default='./tmp/data/ml1m-mf-processed_ui_history.dat', 
                      help='path for cache the filtered data')
    data.add_argument('--reconstruct-data', default=True, help='re-filter the data')
    data.add_argument('--train-test-split', default='loo', help='train/test split method')
    data.add_argument('--random-split', default=False, help='random split or according to time')
    data.add_argument('--train_test-freq-bd', help='split the data freq-wise, bound of the user freq')
    data.add_argument('--train-valid-freq-bd', help='split the data freq-wise, bound of the user freq')
    data.add_argument('--test-latest-n', default=1)
    data.add_argument('--valid-latest-n', default=1)
    data.add_argument('--test-ratio', default=0.01)
    data.add_argument('--valid-ratio', default=0.01)
    data.add_argument('--num-negatives', default=1)
    data.add_argument('--batch-size-train', default=1024)
    data.add_argument('--batch-size-valid', default=1024)
    data.add_argument('--batch-size-test', default=1024)
    data.add_argument('--multi-cpu-train', default=False)  # TODO
    data.add_argument('--multi-cpu-valid', default=False)
    data.add_argument('--multi-cpu-test', default=False)
    data.add_argument('--num-workers-train', default=1)   # TODO
    data.add_argument('--num-workers-valid', default=1)
    data.add_argument('--num-workers-test', default=1)
    data.add_argument('--device-ids-test', default=[0, 1, 2, 3], help='devices used for multi-processing evaluate')

    regularize.add_argument('--regularizer', default='adaptive', help='type of regularizer, fixed or adaptive')
    regularize.add_argument('--penalty-param-path', default=None, help='save lambda for analysis')
    regularize.add_argument('--lambda-network-grad-clip', default=100)
    regularize.add_argument('--lambda-network-type', default='dimension-wise', help='granularity of lambda')
    regularize.add_argument('--lambda-network-lr', default=0.1, help='lr for lambda update')
    regularize.add_argument('--lambda-network-optimizer', default='adam')
    regularize.add_argument('--lambda-network-dp-prob', default=0.5)
    regularize.add_argument('--lambda-network-multi-step', default=1)
    regularize.add_argument('--fixed-lambda-candidate',
                            default=[10, 1, 0, 0.1, 0.01, 1e-3, 1e-4, 1e-5],
                            help='Grid search candidate for fixed')
    regularize.add_argument('--max-steps', default=1e8)
    regularize.add_argument('--lambda-update-interval', default=1,
                            help='Interval between two lambda updates')
    regularize.add_argument('--use-cuda', default=True)
    regularize.add_argument('--device-id', default=1, help='Training Devices')

    factorize.add_argument('--factorizer', default='mf', help='Type of the Factorization Model')
    factorize.add_argument('--metric_topk', default=10, help='Top K for HR and NDCG metric')
    factorize.add_argument('--latent-dim', default=8)

    type_opt = 'mf'
    matrix_factorize.add_argument('--{}-optimizer'.format(type_opt), default='adam')
    matrix_factorize.add_argument('--{}-lr'.format(type_opt), default=1e-3)
    matrix_factorize.add_argument('--{}-grad-clip'.format(type_opt), default=1)

    log.add_argument('--log-interval', default=1)
    log.add_argument('--tensorboard', default='./tmp/runs')
    return parser


class Engine(object):
    """Engine wrapping the training & evaluation
       of adpative regularized maxtirx factorization
    """
    def __init__(self, opt):
        self._opt = opt
        self._sampler = setup_sample_generator(opt)

        self._opt['num_users'] = self._sampler.num_users
        self._opt['num_items'] = self._sampler.num_items
        # self._opt['max_steps'] = 200
        self._opt['eval_res_path'] = self._opt['eval_res_path'].format(alias=self._opt['alias'],
                                                                       epoch_idx='{epoch_idx}')
        if self._opt['penalty_param_path'] is not None:
            self._opt['penalty_param_path'] = self._opt['penalty_param_path'].format(
                alias=self._opt['alias'],                                                                 epoch_idx='{epoch_idx}')
        self._factorizer = setup_factorizer(opt)
        self._factorizer_assumed = setup_factorizer(opt)
        self._regularizer = setup_regularizer(opt)
        self._mode = None

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        assert new_mode in ['complete', 'partial', None]  # training a complete trajectory or a partial trajctory
        self._mode = new_mode

    def train_fixed_reg(self):
        self.mode = 'complete'
        while self._regularizer.set_cur_lambda():
            valid_metrics, _ = self.train_an_episode(max_steps=self._opt['max_steps'])
            self._regularizer.track_metrics(valid_metrics)

    def train_alter_reg(self):
        self.mode = 'complete'
        self.train_an_episode(self._opt['max_steps'])

    # 差分进化算法
    def GenerateTrainVector(self, ID, maxID, lr_min, lr_max, reg_rate_min, reg_rate_max, lr_matrix, reg_rate_matrix):
        SFGSS = 8
        SFHC = 20
        Fl = 0.1
        Fu = 0.9
        tuo1 = 0.1
        tuo2 = 0.03
        tuo3 = 0.07

        Result = np.empty(shape=(2, 1))

        u1 = ID
        u2 = ID
        u3 = ID
        while u1 == ID:
            u1 = np.random.randint(0, maxID)
        while (u2 == ID) or (u2 == u1):
            u2 = np.random.randint(0, maxID)
        while (u3 == ID) or (u3 == u2) or (u3 == u1):
            u3 = np.random.randint(0, maxID)

        rand1 = np.random.rand()
        rand2 = np.random.rand()
        rand3 = np.random.rand()
        F = np.random.rand()
        K = np.random.rand()

        if rand3 < tuo2:
            F = SFGSS
        elif tuo2 <= rand3 < tuo3:
            F = SFHC
        elif rand2 < tuo1 and rand3 > tuo3:
            F = Fl + Fu * rand1

        temp1 = lr_matrix[u2][0] - lr_matrix[u3][0]
        temp2 = temp1 * F
        temp_mutation = lr_matrix[u1][0] + temp2
        temp1 = temp_mutation - lr_matrix[ID][0]
        temp2 = temp1 * K
        Result[0][0] = lr_matrix[ID][0] + temp2

        temp1 = reg_rate_matrix[u2][0] - reg_rate_matrix[u3][0]
        temp2 = temp1 * F
        temp_mutation = reg_rate_matrix[u1][0] + temp2
        temp1 = temp_mutation - reg_rate_matrix[ID][0]
        temp2 = temp1 * K
        Result[1][0] = reg_rate_matrix[ID][0] + temp2

        if Result[0][0] <= lr_min:
            Result[0][0] = lr_min
        if Result[0][0] >= lr_max:
            Result[0][0] = lr_max
        if Result[1][0] <= reg_rate_min:
            Result[1][0] = reg_rate_min
        if Result[1][0] >= reg_rate_max:
            Result[1][0] = reg_rate_max

        return Result




    def train_an_episode(self, max_steps, episode_idx=''):
        """Train a regularized matrix factorization model"""
        assert self.mode in ['partial', 'complete']

        print('-' * 80)
        print('[{} episode {} starts!]'.format(self.mode, episode_idx))

        log_interval = self._opt.get('log_interval')
        eval_interval = self._opt.get('eval_interval')
        lambda_update_interval = self._opt.get('lambda_update_interval')

        status = dict()
        print('Initializing ...')
        self._regularizer.init_episode()
        self._factorizer.init_episode()
        curr_lambda = self._regularizer.init_lambda()
        valid_mf_loss, train_mf_loss = np.inf, np.inf
        epoch_start = datetime.now()

        lr_min = 0
        lr_max = 0.01
        reg_rate_min = 0
        reg_rate_max = 0.01
        individual_num = 5
        max_ndcg = 0

        lr_matrix = np.empty(shape=(individual_num, 1))
        reg_rate_matrix = np.empty(shape=(individual_num, 1))

        for i in range(individual_num):
            xx = lr_min + np.random.rand() * (lr_max - lr_min)
            yy = reg_rate_min + np.random.rand() * (reg_rate_max - reg_rate_min)
            lr_matrix[i][0] = xx
            reg_rate_matrix[i][0] = yy

        for step_idx in range(int(max_steps)):  # TODO: terminal condition for an episode

            if ((step_idx % self._sampler.num_batches_train) % 600 == 0):
                model = self._factorizer.model
                model_name = self._opt['factorizer']
                epoch_idx = int(step_idx / self._sampler.num_batches_train)
                eval_res_path = self._opt['eval_res_path'].format(epoch_idx=epoch_idx)
                ui_uj_df = self._sampler.test_ui_uj_df
                item_pool = self._sampler.item_pool
                top_k = self._opt['metric_topk']
                use_cuda = self._opt['use_cuda']
                device_ids = self._opt['device_ids_test']
                num_workers = self._opt['num_workers_test']

                test_metrics = evaluate_ui_uj_df(model=model, model_name=model_name, card_feat=None,
                                                 ui_uj_df=ui_uj_df, item_pool=item_pool,
                                                 metron_top_k=top_k, eval_res_path=eval_res_path,
                                                 use_cuda=use_cuda, device_ids=device_ids, num_workers=num_workers)

                ndcg_no_de = test_metrics['ndcg']
                recall_no_de = test_metrics['hr']
                auc_no_de = test_metrics['auc']
                for ID in range(individual_num):
                    evolution = self.GenerateTrainVector(ID, individual_num, lr_min, lr_max, reg_rate_min, reg_rate_max, lr_matrix, reg_rate_matrix)
                    lr = evolution[0][0]
                    reg_rate = evolution[1][0]

                    # Prepare status for current step
                    status['done'] = False
                    status['sampler'] = self._sampler
                    self._factorizer_assumed.copy(self._factorizer) # TODO move this to regularizer???
                    status['factorizer'] = self._factorizer_assumed  # for assumed copy

                    self._opt['mf_lr'] = lr
                    self._opt['fixed_lambda_candidate'] = reg_rate

                    curr_lambda = self._regularizer.get_lambda(status=status)

                    valid_mf_loss = self._regularizer.valid_mf_loss

                    train_mf_loss = self._factorizer.update(self._sampler, l2_lambda=curr_lambda)
                    train_l2_penalty = self._factorizer.l2_penalty
                    status['train_mf_loss'] = train_mf_loss

                    # Logging & Evaluate on the Evaluate Set
                    if self.mode == 'complete' and step_idx % log_interval == 0:
                        epoch_idx = int(step_idx / self._sampler.num_batches_train)
                        # for analysis, comment it to accelerate training
                        mf_grad_norm = self._factorizer.get_grad_norm()
                        if hasattr(self._regularizer, 'get_grad_norm'):
                            reg_grad_norm = self._regularizer.get_grad_norm()
                        else:
                            reg_grad_norm = 0
                        if ((step_idx % self._sampler.num_batches_train) % 600 == 0):
                            print("[Epoch: {}] ||  mf_loss: {} || mf_grad_norm: {}".format(epoch_idx, train_mf_loss, mf_grad_norm))

                        user_lambda, item_lambda = curr_lambda.cpu().numpy(), curr_lambda.cpu().numpy()
                    model = self._factorizer.model
                    model_name = self._opt['factorizer']
                    epoch_idx = int(step_idx / self._sampler.num_batches_train)
                    eval_res_path = self._opt['eval_res_path'].format(epoch_idx=epoch_idx)
                    ui_uj_df = self._sampler.test_ui_uj_df
                    item_pool = self._sampler.item_pool
                    top_k = self._opt['metric_topk']
                    use_cuda = self._opt['use_cuda']
                    device_ids = self._opt['device_ids_test']
                    num_workers = self._opt['num_workers_test']

                    test_metrics = evaluate_ui_uj_df(model=model, model_name=model_name, card_feat=None,
                                                     ui_uj_df=ui_uj_df, item_pool=item_pool,
                                                     metron_top_k=top_k, eval_res_path=eval_res_path,
                                                     use_cuda=use_cuda, device_ids=device_ids, num_workers=num_workers)
                    ndcg_de = test_metrics['ndcg']
                    recall_de = test_metrics['hr']
                    auc_de = test_metrics['auc']

                    if ndcg_de >= ndcg_no_de:
                        lr_matrix[ID][0] = evolution[0][0]
                        reg_rate_matrix[ID][0] = evolution[1][0]
                        ndcg_no_de = ndcg_de
                        recall_no_de = recall_de
                        auc_no_de = auc_de
                    print('auc', auc_de)
                    print('hr', recall_de)
                    print('ndcg', ndcg_de)
            else:
                # Prepare status for current step
                status['done'] = False
                status['sampler'] = self._sampler
                self._factorizer_assumed.copy(self._factorizer) # TODO move this to regularizer???
                status['factorizer'] = self._factorizer_assumed  # for assumed copy

                self._opt['mf_lr'] = lr
                self._opt['fixed_lambda_candidate'] = reg_rate

                curr_lambda = self._regularizer.get_lambda(status=status)

                valid_mf_loss = self._regularizer.valid_mf_loss

                train_mf_loss = self._factorizer.update(self._sampler, l2_lambda=curr_lambda)
                train_l2_penalty = self._factorizer.l2_penalty
                status['train_mf_loss'] = train_mf_loss

                # Logging & Evaluate on the Evaluate Set
                if self.mode == 'complete' and step_idx % log_interval == 0:
                    epoch_idx = int(step_idx / self._sampler.num_batches_train)
                    # for analysis, comment it to accelerate training
                    mf_grad_norm = self._factorizer.get_grad_norm()
                    if hasattr(self._regularizer, 'get_grad_norm'):
                        reg_grad_norm = self._regularizer.get_grad_norm()
                    else:
                        reg_grad_norm = 0
                    if ((step_idx % self._sampler.num_batches_train) % 600 == 0):
                        print("[Epoch: {}] ||  mf_loss: {} || mf_grad_norm: {}".format(epoch_idx, train_mf_loss, mf_grad_norm))

                    user_lambda, item_lambda = curr_lambda.cpu().numpy(), curr_lambda.cpu().numpy()

            if (step_idx % self._sampler.num_batches_train == 0) and (epoch_idx % 1 == 0):
                print('Evaluate on test ...')
                start = datetime.now()
                eval_res_path = self._opt['eval_res_path'].format(epoch_idx=epoch_idx)
                eval_res_dir, _ = os.path.split(eval_res_path)
                if not os.path.exists(eval_res_dir):
                    os.mkdir(eval_res_dir)
                model = self._factorizer.model
                model_name = self._opt['factorizer']


                ui_uj_df = self._sampler.test_ui_uj_df
                item_pool = self._sampler.item_pool
                top_k = self._opt['metric_topk']
                use_cuda = self._opt['use_cuda']
                device_ids = self._opt['device_ids_test']
                num_workers = self._opt['num_workers_test']

                test_metrics = evaluate_ui_uj_df(model=model, model_name=model_name, card_feat=None,
                                      ui_uj_df=ui_uj_df, item_pool=item_pool,
                                      metron_top_k=top_k, eval_res_path=eval_res_path,
                                      use_cuda=use_cuda, device_ids=device_ids, num_workers=num_workers)
                # save lambda network's parameter
                print("=================================================================")
                print("AUC:", test_metrics['auc'])
                print("recall@{}: {}".format(top_k, test_metrics['hr']))
                print("ndcg@{}: {}".format(top_k, test_metrics['ndcg']))
                print("=================================================================")

                end = datetime.now()
                print('Evaluate Time {} minutes'.format((end - start).total_seconds() / 60))
                epoch_end = datetime.now()
                dur = (epoch_end - epoch_start).total_seconds() / 60
                epoch_start = datetime.now()
                print('[Epoch {:4d}] train MF loss: {:04.8f}, '
                      'valid loss: {:04.8f}, time {:04.8f} minutes'.format(epoch_idx,
                                                                           train_mf_loss,
                                                                           valid_mf_loss,
                                                                           dur))

    def train(self):
        if self._opt['regularizer'] == 'fixed':
            self.train_fixed_reg()
        elif 'alter' in self._opt['regularizer']:
            self.train_alter_reg()
