
from datetime import datetime
import logging
from functools import partial
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import FakedditData, fakeddit_custom_collate_fn,FakedditDatabase
import random
import numpy as np
import os
from utils import metrics
BLUE = '\033[94m'
ENDC = '\033[0m'


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

    logger.info(BLUE + 'Gpu: ' + ENDC + f"{args.gpu} ")

    logger.info(BLUE + 'Model: ' + ENDC + f"{args.model_path} ")

    logger.info(BLUE + "Dataset: " + ENDC + f"{args.dataset_id}")

    logger.info(BLUE + "Metric: " + ENDC + f"{args.metric}")

    logger.info(BLUE + "Number of retrieved items used in this testing: " + ENDC + f"{args.num_of_retrieved_items}")
    logger.info(BLUE + "Number of retrieved users used in this testing: " + ENDC + f"{args.num_of_retrieved_users}")

    logger.info(BLUE + "Alpha: " + ENDC + f"{args.alpha}")

    logger.info(BLUE + "Number of frames: " + ENDC + f"{args.frame_num}")

    logger.info(BLUE + "Testing Starts!" + ENDC)


def delete_special_tokens(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:

        content = file.read()

    content = content.replace(BLUE, '')

    content = content.replace(ENDC, '')

    with open(file_path, 'w', encoding='utf-8') as file:

        file.write(content)


def test(args,database=None):

    model_id = args.model_id

    dataset_id = args.dataset_id

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder_name = f"test_{model_id}_{dataset_id}_{timestamp}"

    father_folder_name = args.save

    if not os.path.exists(father_folder_name):

        os.makedirs(father_folder_name)

    folder_path = os.path.join(father_folder_name, folder_name)

    os.mkdir(folder_path)

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

    batch_size = args.batch_size

    ds = FakedditData
    custom_collate_fn = fakeddit_custom_collate_fn
    db=FakedditDatabase

    if database is None:
        database = db(os.path.join(args.dataset_path, args.dataset_id))
    test_data = ds(os.path.join(os.path.join(args.dataset_path, args.dataset_id, 'test.pkl')),database)

    custom_collate_fn_partial = partial(custom_collate_fn, num_of_retrieved_items=args.num_of_retrieved_items,num_of_retrieved_users=args.num_of_retrieved_users,
                                        num_of_frames=args.frame_num)

    test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, collate_fn=custom_collate_fn_partial)

    model = torch.load(args.model_path)



    print_init_msg(logger, args)

    model.eval()
    pred_list = []
    label_list = []

    with torch.no_grad():

        for batch in tqdm(test_data_loader, desc='Testing'):

            batch = [item.cuda() if isinstance(item, torch.Tensor) else item for item in batch]



            visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding, \
                    retrieved_textual_feature_embedding, retrieved_label, user, retrieved_user, retrieved_user_similarity, label = batch

            output, loss_other = model.forward(visual_feature_embedding, textual_feature_embedding, similarity,
                                               retrieved_visual_feature_embedding,
                                               retrieved_textual_feature_embedding, retrieved_label, user,
                                               retrieved_user, retrieved_user_similarity)



            label_list.extend(label.squeeze(1).detach().cpu().numpy().tolist())
            pred_list.extend(output.squeeze(1).detach().cpu().numpy().tolist())

    for i in range(len(label_list)):
        if pred_list[i] > 0.5:
            pred_list[i] = 1
        else:
            pred_list[i] = 0

    metric_res=metrics(label_list, pred_list)

    logger.info(f"[ Test Result ]:  \n {str(metric_res)}")
    logger.info("Test is ended!")

    delete_special_tokens(f"{father_folder_name}/{folder_name}/log.txt")
    return metric_res


def main(args):

    seed_init(args.seed)

    test(args)

