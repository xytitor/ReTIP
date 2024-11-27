import logging
import sys
from datetime import datetime
from argument import train_argument_init
import time
import numpy as np
from tqdm import tqdm
import os
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from dataset import FakedditData, fakeddit_custom_collate_fn, FakedditDatabase
from retip import Model

import random
from functools import partial
from test import test


BLUE = '\033[94m'
ENDC = '\033[0m'
RED = '\033[91m'


def seed_init(seed):
    seed = int(seed)

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True


def print_init_msg(logger, args):
    logger.info(BLUE + 'Random Seed: ' + ENDC + f"{args.seed} ")

    logger.info(BLUE + 'Device: ' + ENDC + f"{args.gpu} ")

    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_id} ")

    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")

    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")

    logger.info(BLUE + "Optimizer: " + ENDC + f"{args.optim}(lr = {args.lr})")

    logger.info(BLUE + "Total Epoch: " + ENDC + f"{args.epochs} Turns")

    logger.info(BLUE + "Early Stop: " + ENDC + f"{args.early_stop_turns} Turns")

    logger.info(BLUE + "Batch Size: " + ENDC + f"{args.batch_size}")

    logger.info(BLUE + "Number of retrieved items used in this training: " + ENDC + f"{args.num_of_retrieved_items}")
    logger.info(BLUE + "Number of retrieved users used in this testing: " + ENDC + f"{args.num_of_retrieved_users}")

    logger.info(BLUE + "Alpha: " + ENDC + f"{args.alpha}")

    logger.info(BLUE + "Number of frames: " + ENDC + f"{args.frame_num}")

    logger.info(BLUE + "Training Starts!" + ENDC)


def make_saving_folder_and_logger(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder_name = f"train_{args.model_id}_{args.dataset_id}_{args.metric}_{timestamp}"

    father_folder_name = args.save

    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)

    os.mkdir(folder_path)

    os.mkdir(os.path.join(folder_path, "trained_model"))

    logger = logging.getLogger()

    logger.handlers = []

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')

    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    return father_folder_name, folder_name, logger


def delete_model(father_folder_name, folder_name, min_turn):
    model_name_list = os.listdir(f"{father_folder_name}/{folder_name}/trained_model")

    for i in range(len(model_name_list)):

        if model_name_list[i] != f'model_{min_turn}.pth':
            os.remove(os.path.join(f'{father_folder_name}/{folder_name}/trained_model', model_name_list[i]))


def force_stop(msg):
    print(msg)

    sys.exit(1)


def delete_special_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content = content.replace(BLUE, '')

    content = content.replace(ENDC, '')

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def train_val(args):
    father_folder_name, folder_name, logger = make_saving_folder_and_logger(args)



    ds = FakedditData
    custom_collate_fn = fakeddit_custom_collate_fn
    db = FakedditDatabase
    custom_collate_fn_partial = partial(custom_collate_fn, num_of_retrieved_items=args.num_of_retrieved_items,
                                        num_of_retrieved_users=args.num_of_retrieved_users,
                                        num_of_frames=args.frame_num)
    database = db(os.path.join(args.dataset_path, args.dataset_id))
    train_data = ds(os.path.join(args.dataset_path, args.dataset_id, 'train.pkl'), database)
    valid_data = ds(os.path.join(os.path.join(args.dataset_path, args.dataset_id, 'valid.pkl')), database)
    graph_metadata = None
    train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=custom_collate_fn_partial)
    valid_data_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, collate_fn=custom_collate_fn_partial)


    model = Model(feature_dim=args.feature_dim, alpha=args.alpha, retrieval_num=args.num_of_retrieved_items,
                  retrieval_user_num=args.num_of_retrieved_users,hypergraph_layer_num=args.hypergraph_layer_num,agg_fun=args.agg_fun)

    model = model.cuda()

    if args.loss == 'BCE':

        loss_fn = torch.nn.BCELoss()

    elif args.loss == 'MSE':

        loss_fn = torch.nn.MSELoss()

    else:

        force_stop('Invalid parameter loss!')

    loss_fn.cuda()

    if args.optim == 'Adam':

        optim = Adam(model.parameters(), args.lr)

    elif args.optim == 'SGD':

        optim = SGD(model.parameters(), args.lr)

    else:

        force_stop('Invalid parameter optim!')

    min_total_valid_loss = 1008611

    min_turn = 0

    print_init_msg(logger, args)
    epoch_times = []
    for i in range(args.epochs):

        logger.info(f"-----------------------------------Epoch {i + 1} Start!-----------------------------------")

        epoch_start_time = time.time()

        min_train_loss, total_valid_loss = run_one_epoch(model, loss_fn, optim, train_data_loader, valid_data_loader,
                                                         args)
        epoch_end_time = time.time()

        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        logger.info(f"[ Epoch {i + 1} (train) ]: training_step_loss = {min_train_loss}")

        logger.info(f"[ Epoch {i + 1} (valid) ]: valid_total_loss = {total_valid_loss}")

        if total_valid_loss < min_total_valid_loss:
            min_total_valid_loss = total_valid_loss

            min_turn = i + 1

        logger.critical(
            f"Current Best Total Loss comes from Epoch {min_turn} , min_total_loss = {min_total_valid_loss}")
        save_path = f"{father_folder_name}/{folder_name}/trained_model/model_{i + 1}.pth"

        torch.save(model, save_path)

        logger.info(f"Model has been saved successfully!")

        if (i + 1) - min_turn > args.early_stop_turns:
            break

    delete_model(father_folder_name, folder_name, min_turn)

    logger.info(BLUE + "Training is ended!" + ENDC)
    min_model_path = f"{father_folder_name}/{folder_name}/trained_model/model_{min_turn}.pth"

    logger.info(
        f"Best Model comes from Epoch {min_turn} , min_total_loss = {min_total_valid_loss},location:{min_model_path}")

    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")

    logger.info(RED + '- -----------------------------------TRAIN END!-----------------------------------' + ENDC)
    logger.info(BLUE + f"Average epoch time:" + ENDC + f"{np.mean(epoch_times)}")

    return min_model_path, database


def run_one_epoch(model, loss_fn, optim, train_data_loader, valid_data_loader, args):
    model.train()

    min_train_loss = 1008611

    for batch in tqdm(train_data_loader, desc='Training Progress'):

        batch = [item.cuda() if isinstance(item, torch.Tensor) else item for item in batch]


        visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding, \
                retrieved_textual_feature_embedding, retrieved_label, user, retrieved_user, retrieved_user_similarity, label = batch

        output, loss_other = model.forward(visual_feature_embedding, textual_feature_embedding, similarity,
                                           retrieved_visual_feature_embedding,
                                           retrieved_textual_feature_embedding, retrieved_label, user,
                                           retrieved_user, retrieved_user_similarity)

        loss = loss_fn(output, label)
        for l_k in loss_other.keys():
            loss += loss_other[l_k]


        optim.zero_grad()

        loss.backward()
        optim.step()
        if min_train_loss > loss:
            min_train_loss = loss

    model.eval()

    total_valid_loss = 0

    with torch.no_grad():

        for batch in tqdm(valid_data_loader, desc='Validating Progress'):


            batch = [item.cuda() if isinstance(item, torch.Tensor) else item for item in batch]


            visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding, \
                    retrieved_textual_feature_embedding, retrieved_label, user, retrieved_user, retrieved_user_similarity, label = batch
            output, loss_other = model.forward(visual_feature_embedding, textual_feature_embedding, similarity,
                                               retrieved_visual_feature_embedding,
                                               retrieved_textual_feature_embedding, retrieved_label, user,
                                               retrieved_user, retrieved_user_similarity)

            loss = loss_fn(output, label)
            for l_k in loss_other.keys():
                loss += loss_other[l_k]
            total_valid_loss += loss

    return min_train_loss, total_valid_loss





def main_local(seed=2024):
    args = train_argument_init()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.wandb = False
    args.batch_size = 64
    args.early_stop_turns = 5
    args.epochs = 100
    args.feature_dim = 512
    args.dataset_id='fakeddit'
    args.lr = 0.0001
    args.num_of_retrieved_items = 9
    args.num_of_retrieved_users = 3
    args.seed = seed
    seed_init(args.seed)
    min_model_path, database = train_val(args)
    args.model_path = min_model_path
    res=test(args, database)
    return res


if __name__ == '__main__':
    main_local()
