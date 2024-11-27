
import argparse

def train_argument_init():


    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=str, default='2024', help='Seed for reproducibility')

    parser.add_argument('--gpu', type=str, default='1', help='Device for training')

    parser.add_argument('--metric', type=str, default='BCE', help='Metric for evaluation')

    parser.add_argument('--save', type=str, default='./train_results', help='Directory to save results')

    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')

    parser.add_argument('--early_stop_turns', type=int, default=20, help='Number of turns for early stopping')

    parser.add_argument('--loss', type=str, default='BCE', help='Loss function for training')

    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer for training')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    parser.add_argument('--dataset_id', type=str, default='fakeddit', help='Dataset identifier')


    parser.add_argument('--dataset_path', type=str, default='', help='Path to the dataset')

    parser.add_argument('--model_id', type=str, default='', help='Model id')

    parser.add_argument('--feature_num', type=int, default=2, help='Number of features')

    parser.add_argument('--num_of_retrieved_items', type=int, default=3,
                        help='Number of retrieved items, hyper-parameter')

    parser.add_argument('--feature_dim', type=int, default=768, help='Dimension of features')

    parser.add_argument('--label_dim', type=int, default=1, help='Dimension of labels')

    parser.add_argument('--alpha', type=float, default=0.6, help='Alpha, hyper-parameter')


    parser.add_argument('--model_path', default='', type=str, help='path of trained model')
    parser.add_argument('--wandb', type=bool, default=True, help='Is open wandb')
    parser.add_argument('--temp_1', type=float, default=True, help='temp_1')
    parser.add_argument('--temp_2', type=float, default=True, help='temp_2')
    parser.add_argument('--temp_3', type=float, default=True, help='temp_3')
    parser.add_argument('--temp_4', type=float, default=True, help='temp_4')
    parser.add_argument('--agg_fun', type=str, default='softmax_then_sum', help='agg function')
    parser.add_argument('--hypergraph_layer_num', type=int, default=1, help='hypergraph layer num')
    args_ = parser.parse_args()
    return args_

def test_argument_init():
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default='2024', type=str, help='value of random seed')

    parser.add_argument('--gpu', default='1', type=str, help='device used in testing')

    parser.add_argument('--metric', default=['BCE'], type=list, help='the judgement of the testing')

    parser.add_argument('--save', default='./test_results', type=str, help='folder to save the results')

    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')

    parser.add_argument('--dataset_id', default='fakeddit', type=str, help='id of dataset')

    parser.add_argument('--dataset_path', default='', type=str, help='path of dataset folder')

    parser.add_argument('--model_id', default='ReTIP', type=str, help='id of model')

    parser.add_argument('--num_of_retrieved_items', default=3, type=int,
                        help='number of retrieved items used this training, hyper-parameter')

    parser.add_argument('--alpha', default=0.6, type=int, help='Alpha, hyper-parameter')


    parser.add_argument('--feature_num', type=int, default=2, help='Number of features')

    parser.add_argument('--feature_dim', type=int, default=512, help='Dimension of features')

    parser.add_argument('--label_dim', type=int, default=1, help='Dimension of labels')
    parser.add_argument('--model_path',
                        default='',
                        type=str, help='path of trained model')
    parser.add_argument('--agg_fun', type=str, default='softmax_then_sum', help='agg function')
    parser.add_argument('--hypergraph_layer_num', type=int, default=1, help='hypergraph layer num')
    args = parser.parse_args()
    return args

