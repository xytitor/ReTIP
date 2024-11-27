import os
from tqdm import tqdm
from torch_geometric.data import Data
tok_k=20
walk_times=100
walk_length=50
LOC=''
user_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/user_mean.pkl')
filtered_comments_save_path = os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/filtered_comments.pkl')
save_path=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/user_retrieval.pt')
node2community_save_path=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/node2community.pt')
import time
import pandas as pd
import torch
import numpy as np
import networkx as nx
import community as community_louvain
import random
def levy_step(mu, size=1):
    return np.random.standard_normal(size) * (np.random.uniform(0, 1, size) ** (-1.0 / mu))
def levy_random_walk(graph, start, walk_length, mu=2.5):


    walks = []

    for start_node in start.tolist():
        current_node = start_node
        walk = [current_node]

        for _ in range(walk_length):
            if current_node not in graph or len(graph[current_node]) == 0:
                break  #


            step_length = levy_step(mu)[0]
            step_length = max(1, int(abs(step_length)))


            for _ in range(step_length):

                neighbors = graph[current_node]
                if len(neighbors) == 0:
                    break
                else:
                    next_node = random.choice(neighbors)
                current_node = next_node
            walk.append(next_node)

        walks.append(torch.tensor(walk))

    return walks

def pyg_to_networkx(data):
    G = nx.Graph()
    for i in tqdm(range(data.num_nodes),total=data.num_nodes):
        G.add_node(i)
    edge_list = data.edge_index.t().cpu().numpy()
    G.add_edges_from(edge_list)
    return G
def get_node2community(graph):
    if os.path.exists(node2community_save_path):
        return torch.load(node2community_save_path)
    G = pyg_to_networkx(graph)

    partition = community_louvain.best_partition(G)

    node2community = {}
    for node, community_id in tqdm(partition.items(), total=len(partition)):
        node2community[node] = community_id
    torch.save(node2community, node2community_save_path)
    return node2community

def generate_complete_graph_edges(node_indices):
    edges = []
    num_nodes = len(node_indices)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):

            edges.append((node_indices[i], node_indices[j]))
            edges.append((node_indices[j], node_indices[i]))


    edge_tensor = torch.tensor(edges, dtype=torch.long).t()

    return edge_tensor

def load_user_graph():
    graph_path=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/user_graph.pt')
    if  not os.path.exists(graph_path):
        user_df=pd.read_pickle(user_df_dir)
        up_user_df=pd.DataFrame()
        up_user_df['user_id']=user_df['user_id']
        user_df.set_index('user_id',inplace=True)
        user_mapping = {index: i for i, index in enumerate(user_df.index.unique())}
        comments_df=pd.read_pickle(filtered_comments_save_path)
        grouped_comments = comments_df.groupby('parent_id')

        edge=[]
        for name, group in tqdm(grouped_comments):
            indices=[user_mapping[k] for k in group['author'].values]
            if len(indices)>1:
                edge.append(generate_complete_graph_edges(indices))
        user_embeddings = np.array(user_df['user_embedding'].tolist())
        num_nodes=len(user_embeddings)

        edge=torch.cat(edge,dim=1).cuda()

        data = Data( edge_index=edge,num_nodes=num_nodes)
        torch.save(data, graph_path)
        return data
    else:
        return torch.load(graph_path)




def walk_graph(graph,node_tensor):
    path_list=[]
    for i in range(walk_times):

        path_list.append(levy_random_walk(graph,node_tensor,walk_length))
    return path_list
def user_retrieval(paths_list,node2com):
    node_num=len(paths_list[0])
    res=[]
    for n in range(node_num):
        counts=[]
        for t in range(walk_times):
            path=paths_list[t][n]
            unique_elements= torch.unique(path)
            community_id = node2com[int(unique_elements[0])]

            mask = unique_elements.new_full(unique_elements.shape, False)
            for i in range(unique_elements.shape[0]):
                if node2com[int(unique_elements[i])] == community_id:
                    mask[i] = True
            mask = mask.bool()
            unique_elements = unique_elements[mask]
            counts.append(unique_elements)

        counts=torch.cat(counts,dim=-1)
        unique_elements, counts = torch.unique(counts, return_counts=True)

        sorted_counts, sorted_indices = torch.sort(counts, descending=True)

        sorted_unique_elements = unique_elements[sorted_indices]

        if sorted_unique_elements.shape[0] < tok_k + 1:
            padding_num = tok_k + 1 - sorted_unique_elements.shape[0]
            padding_elements = torch.full((padding_num,), sorted_unique_elements[0])
            padding_count = torch.full((padding_num,), 1)
            sorted_unique_elements = torch.cat((sorted_unique_elements[1:], padding_elements), dim=-1)
            sorted_counts = torch.cat((sorted_counts[1:], padding_count), dim=-1)
        else:
            sorted_unique_elements = sorted_unique_elements[1:tok_k + 1]
            sorted_counts = sorted_counts[1:tok_k + 1]
        item = torch.stack([sorted_unique_elements, sorted_counts])
        res.append(item)
    return torch.stack(res)

def main():
    graph=load_user_graph()
    num_nodes = graph.num_nodes
    num_edges = graph.edge_index.size(1)
    print(f"node num：{num_nodes}")
    print(f"edge num：{num_edges}")
    edge=graph.edge_index
    edge = torch.cat([edge, edge.flip(0)], dim=1)
    edge = edge.t()
    edge = torch.unique(edge, dim=0)
    edge = edge.t()
    mask = edge[0] != edge[1]
    edge = edge[:, mask]
    graph.edge_index=edge

    print('node nums:',graph.num_nodes )
    print('edge nums:', edge.shape[1])
    self_loops = (edge[0] == edge[1]).sum().item()
    print(f"self loops: {self_loops}")
    node2com = get_node2community(graph)
    num_nodes = graph.num_nodes
    print(num_nodes)
    graph_set = {}
    for r, c in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()):
        if r not in graph_set:
            graph_set[r] = []
        graph_set[r].append(c)
    del graph

    node_tensor = torch.arange(0, num_nodes).cuda()
    node_chunks = torch.chunk(node_tensor, 300)
    res=[]
    for i, chunk in tqdm(enumerate(node_chunks), total=len(node_chunks)):
        paths_list=walk_graph(graph_set,chunk)
        res_tensor=user_retrieval(paths_list,node2com)
        res.append(res_tensor)
    res=torch.cat(res,dim=0)
    torch.save(res, save_path)
    print('')

if __name__ == '__main__':
    main()


