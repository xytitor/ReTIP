import torch.utils.data
import pandas as pd
import numpy as np
import os
import torch
def fakeddit_custom_collate_fn(batch, num_of_retrieved_items,num_of_retrieved_users, num_of_frames):
    top_k_user_num=num_of_retrieved_users
    visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding, \
        retrieved_textual_feature_embedding, retrieved_label, item_id, \
        user,retrieved_user, retrieved_user_similarity,\
        retrieved_item_id, label = zip(*batch)

    visual_feature_embedding = np.array(visual_feature_embedding)


    return ((torch.tensor(visual_feature_embedding, dtype=torch.float)), \
        torch.tensor(textual_feature_embedding, dtype=torch.float).unsqueeze(1), \
        (torch.tensor(similarity))[:, :num_of_retrieved_items], \
        (torch.stack(retrieved_visual_feature_embedding))[:, :num_of_retrieved_items,
        :num_of_frames, :], \
        (torch.stack(retrieved_textual_feature_embedding))[:, :num_of_retrieved_items, :], \
        (torch.tensor(retrieved_label, dtype=torch.float).unsqueeze(2))[:, :num_of_retrieved_items,
        :],

            torch.stack(user),  torch.stack(retrieved_user)[:,:top_k_user_num,:],  torch.stack(retrieved_user_similarity)[:,:top_k_user_num],
        torch.tensor(
        label, dtype=torch.float).unsqueeze(1))


class FakedditData(torch.utils.data.Dataset):

    def __init__(self, path, database):
        super().__init__()

        self.path = path
        self.graph_path = os.path.dirname(path) + '/graph/'
        self.dataframe = pd.read_pickle(path)


        self.database = database
        self.visual_feature_embedding_list = self.dataframe['visual_feature_embedding'].tolist()
        self.textual_feature_embedding_list = self.dataframe['textual_feature_embedding'].tolist()
        self.similarity_list = self.dataframe['retrieved_item_similarity_list'].tolist()
        self.retrieved_indices_list = self.dataframe['retrieved_indices_list'].tolist()
        self.retrieved_item_id_list = self.dataframe['retrieved_item_id_list'].tolist()
        self.retrieved_label_list = self.dataframe['retrieved_label_list'].tolist()
        self.label_list = self.dataframe['label'].tolist()
        self.user_id=self.dataframe['author'].tolist()
        self.item_id_list = self.dataframe['id'].tolist()
        self.len = len(self.dataframe)
        del self.dataframe


    def __getitem__(self, index):
        visual_feature_embedding = self.visual_feature_embedding_list[index]

        textual_feature_embedding = self.textual_feature_embedding_list[index]
        similarity = self.similarity_list[index]
        retrieved_indices = self.retrieved_indices_list[index]
        retrieved_label = self.retrieved_label_list[index]
        label = self.label_list[index]
        retrieved_visual_feature_embedding = self.database.visual[retrieved_indices]
        retrieved_textual_feature_embedding = self.database.textual[retrieved_indices]
        item_id = self.item_id_list[index]
        retrieved_item_id = self.retrieved_item_id_list[index]



        uid=self.user_id[index]



        uidx=self.database.user_mapping[uid]
        u=self.database.related_user[uidx]
        retrieved_user_similarity=(u[1])
        retrieved_user=(u[0])
        retrieved_user=torch.stack([self.database.user[r_uidx] for r_uidx in retrieved_user])
        user=self.database.user[uidx]

        return (visual_feature_embedding, textual_feature_embedding, similarity, retrieved_visual_feature_embedding,
                    retrieved_textual_feature_embedding, retrieved_label, item_id,
                    user,retrieved_user,retrieved_user_similarity,
                    retrieved_item_id,label)

    def __len__(self):
        return self.len


class FakedditDatabase(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.graph_path = path + '/graph/'
        self.visual = torch.load(path + '/database_visual_tensor.pt').cuda()
        print('visual loaded!')
        self.textual = torch.load(path + '/database_textual_tensor.pt').cuda()
        print('textual loaded!')
        self.related_user= torch.load(path + '/user_retrieval.pt').cuda()
        print('related_user loaded!')
        user_df= pd.read_pickle(path + '/user_mean.pkl')
        print('user loaded!')
        user_df.set_index('user_id', inplace=True)
        self.user=user_df['user_embedding']
        self.user_mapping = {index: i for i, index in enumerate(user_df.index.unique())}
        self.user=torch.tensor(np.array(self.user.tolist()),dtype=torch.float).cuda()
