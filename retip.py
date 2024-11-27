import torch
import torch.nn as nn
from .HGNNP import HGNNPConv, MultiLayerHGNN
import dhg
class Model(nn.Module):

    def __init__(self, alpha,  feature_dim=768, retrieval_user_num=10,
                 retrieval_num=10, hypergraph_layer_num=1, agg_fun='softmax_then_sum'):
        super(Model, self).__init__()
        self.alpha = alpha
        self.retrieval_num = retrieval_num
        self.retrieval_user_num = retrieval_user_num
        self.feature_dim = feature_dim
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.proj_dim = 500
        self.tanh = nn.Tanh()
        self.visual_embedding = nn.Linear(feature_dim, self.proj_dim)

        self.textual_embedding = nn.Linear(feature_dim, self.proj_dim)

        self.user_embedding = nn.Linear(feature_dim, self.proj_dim)

        self.retrieval_visual_embedding = nn.Linear(feature_dim, self.proj_dim)
        self.retrieval_textual_embedding = nn.Linear(feature_dim, self.proj_dim)
        self.retrieval_user_embedding = nn.Linear(feature_dim, self.proj_dim)

        self.z_dim = 300
        self.HGNNP=MultiLayerHGNN(in_channels=self.proj_dim,out_channels=self.z_dim,hidden_channels=self.proj_dim//2,layer_num=hypergraph_layer_num,agg_fun=agg_fun)

        self.label_embedding_linear = nn.Linear(1, self.z_dim)

        self.hyper_edge_group_num = 9
        self.hyper_edge_index = None
        self.hegw = nn.Parameter(torch.ones(self.hyper_edge_group_num))
        self.prediction_module = nn.Sequential(
            nn.Linear(self.z_dim * 7, 800),
            nn.ReLU(),
            nn.Linear(800, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, visual_feature, textual_feature, similarity, retrieved_visual_feature, retrieved_textual_feature,
                retrieved_label, user, retrieved_user, retrieved_user_similarity, label=None):

        retrieved_visual_feature = retrieved_visual_feature.squeeze(2)
        user = user.unsqueeze(1)


        bs = visual_feature.shape[0]

        visual_feature_emb = self.visual_embedding(visual_feature)

        visual_feature_emb = self.tanh(visual_feature_emb)
        textual_feature_emb = self.textual_embedding(textual_feature)

        textual_feature_emb = self.tanh(textual_feature_emb)

        user_feature_emb = self.user_embedding(user)

        user_feature_emb = self.tanh(user_feature_emb)

        retrieved_textual_feature = self.retrieval_textual_embedding(retrieved_textual_feature)
        retrieved_textual_feature = self.tanh(retrieved_textual_feature)

        retrieved_visual_feature = self.retrieval_visual_embedding(retrieved_visual_feature)
        retrieved_visual_feature = self.tanh(retrieved_visual_feature)

        retrieved_user_feature = self.retrieval_user_embedding(retrieved_user)
        retrieved_user_feature = self.tanh(retrieved_user_feature)

        #
        X = torch.cat([textual_feature_emb, visual_feature_emb, user_feature_emb, retrieved_textual_feature,
                       retrieved_visual_feature, retrieved_user_feature], dim=1)
        X = X.view(-1, X.size(-1))

        g_idx, graph_group_weigh = self.get_hyper_edge_index_and_weight(bs)
        hg = dhg.Hypergraph(X.size(0), g_idx)
        X = self.HGNNP(X, hg, graph_group_weigh)
        X = X.view(bs, -1, X.size(-1))
        retrieved_visual_aggregated_feature, retrieved_textual_aggregated_feature, retrieved_user_aggregated_feature, retrieved_aggregated_label = (
            self.retrieval_aggregation(similarity, X[:, 3:3 + self.retrieval_num, :],
                                       X[:, 3 + self.retrieval_num:3 + self.retrieval_num * 2, :], X[:,
                                                                                                   3 + self.retrieval_num * 2:3 + self.retrieval_num * 2 + self.retrieval_user_num,
                                                                                                   :],
                                       retrieved_user_similarity, retrieved_label))
        retrieved_aggregated_label_embedding = self.label_embedding_linear(retrieved_aggregated_label)

        retrieved_aggregated_label_embedding = self.relu(retrieved_aggregated_label_embedding)

        output = self.prediction_module(

            torch.cat([
                X[:, 0, :], X[:, 1, :], X[:, 2, :], retrieved_visual_aggregated_feature.squeeze(1),
                retrieved_textual_aggregated_feature.squeeze(1), retrieved_user_aggregated_feature.squeeze(1),
                retrieved_aggregated_label_embedding.squeeze(1)
            ], dim=1))



        return output, {


        }

    @staticmethod
    def gen_bias_list(m, n, bias):
        return list(range(m + bias, n + bias))



    def get_hyper_edge_index_and_weight(self, bs):
        hy_idx = []
        for i in range(bs):
            bias = i * (3 + self.retrieval_num * 2 + self.retrieval_user_num)
            idx = self.gen_bias_list(0, 3, bias)
            hy_idx.append(idx)
            #
            retrieval_visual_start_idx = 3 + self.retrieval_num  # 13
            retrieval_user_start_idx = 3 + self.retrieval_num * 2  # 23
            retrieval_user_end_idx = 3 + self.retrieval_num * 2 + self.retrieval_user_num  # 33

            idx = self.gen_bias_list(3, retrieval_visual_start_idx, bias)
            idx.append(0 + bias)
            hy_idx.append(idx)

            idx = self.gen_bias_list(retrieval_visual_start_idx, retrieval_user_start_idx, bias)
            idx.append(1 + bias)
            hy_idx.append(idx)

            idx = self.gen_bias_list(retrieval_user_start_idx, retrieval_user_end_idx, bias)
            idx.append(2 + bias)
            hy_idx.append(idx)

            for i in range(self.retrieval_num):
                hy_idx.append([3 + i, self.retrieval_num + 3 + i])

        return hy_idx, None

    def retrieval_aggregation(self, similarity, retrieved_textual_feature, retrieved_visual_feature,
                              retrieved_user_feature, user_similarity, retrieved_label):

        similarity = torch.softmax(similarity, dim=1)

        similarity = similarity.unsqueeze(2)
        user_similarity = torch.softmax(user_similarity.float(), dim=1)

        user_similarity = user_similarity.unsqueeze(2)
        retrieved_aggregated_label = torch.matmul(similarity.transpose(1, 2), retrieved_label)

        retrieved_textual_aggregated_feature = torch.matmul(similarity.transpose(1, 2), retrieved_textual_feature)
        retrieved_visual_aggregated_feature = torch.matmul(similarity.transpose(1, 2), retrieved_visual_feature)
        retrieved_user_aggregated_feature = torch.matmul(user_similarity.transpose(1, 2), retrieved_user_feature)

        return retrieved_visual_aggregated_feature, retrieved_textual_aggregated_feature, retrieved_user_aggregated_feature, retrieved_aggregated_label
