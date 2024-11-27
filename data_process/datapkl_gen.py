from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import torch
LOC=''
base_dir = LOC + 'DATA_SET/mm_fnd/fakeddit/'

TEXT_EMD_FLAG=True
VISUAL_EMD_FLAG=True
SAVE_FLAG=True
LOAD_FLAG=False
RETRIEVAL_FLAG=False

USE_CLIP=True

from transformers import AutoProcessor, CLIPModel, AutoTokenizer, CLIPTextModel, CLIPTextConfig,CLIPConfig,CLIPVisionConfig
def load_clip_model():
    # text_configuration = CLIPTextConfig(max_position_embeddings=170)
    # vision_configuration = CLIPVisionConfig()
    # configuration = CLIPConfig.from_text_vision_configs(text_config=text_configuration, vision_config=vision_configuration)
    model = CLIPModel.from_pretrained(LOC + 'PRETRAINED_MODEL/' + "clip-vit-base-patch32")

    processor = AutoProcessor.from_pretrained(LOC + 'PRETRAINED_MODEL/' + "clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained(LOC + 'PRETRAINED_MODEL/' + "clip-vit-base-patch32")
    return model, processor,tokenizer
def clip_textual_feature_extraction(tokenizer, model, text):
    inputs = tokenizer([text], padding=True, return_tensors="pt",max_length=77, truncation=True)

    emd=model.get_text_features(**inputs)
    return emd.cpu().detach().numpy().tolist()[0]


def clip_visual_feature_extraction(processor, model, images_path):
    images = []
    for image_path in images_path:
        if not os.path.exists(image_path):
            print('image not found:', image_path)
            continue
        else:
            image = Image.open(image_path)
            images.append(image)
    for i in range(len(images)):
        if images[i].mode != 'RGB':
            images[i] = images[i].convert('RGB')
    if len(images) == 0:
        return None
    inputs = processor(images=images, return_tensors="pt")
    image_features = model.get_image_features(**inputs)

    return image_features.cpu().detach().numpy()



from PIL import Image

def compute_similarity_and_take_top_k(vector1, vector2,k):
    similarity = torch.nn.functional.cosine_similarity(torch.tensor(vector1).cuda(), vector2)
    similarity_list,indices=torch.sort(similarity,descending=True)
    similarity_list=similarity_list[1:k+1].cpu().tolist()
    indices=indices[1:k+1].cpu().tolist()
    return similarity_list,indices
def sort_and_take_top_k(id_list, similarity_list, k):
    zipped_lists = list(zip(similarity_list, id_list))
    sorted_lists = sorted(zipped_lists, key=lambda x: x[0], reverse=True)
    similarity_list, id_list = zip(*sorted_lists)
    similarity_list = similarity_list[1:k+1]
    id_list = id_list[1:k+1]
    return list(id_list), list(similarity_list)

def save_pkl(res_pkl,base_dir):
    for k,v in res_pkl.items():
        res_pkl[k]=v.dropna()
        res_pkl[k].to_pickle(base_dir+k)
    print('done save '+base_dir+k)
def load_pkl(base_dir):
    res_pkl={}
    for i in proceess_data_list:
        res_pkl[i]=pd.read_pickle(base_dir+i)
    for k in res_pkl.keys():
        res_pkl[k]=res_pkl[k].reset_index(drop=True)
    return res_pkl

validate_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/multimodal_only_samples/multimodal_valid.tsv')
test_pubblic_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/multimodal_only_samples/multimodal_test_public.tsv')
train_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/multimodal_only_samples/multimodal_train.tsv')
comments_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/all_comments.tsv')
images_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/public_image_set/')
updated_user_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/user_embeddings.pkl')
if not LOAD_FLAG:
    validate_df=pd.read_csv(validate_df_dir, sep='\t')
    test_pubblic_df=pd.read_csv(test_pubblic_df_dir, sep='\t')
    train_df=pd.read_csv(train_df_dir, sep='\t')
proceess_data_list=['test.pkl','train.pkl','valid.pkl']

res_pkl={}
if not LOAD_FLAG:
    for i in proceess_data_list:
        res_df=pd.DataFrame()
        if i=='train.pkl':
            res_df['text']=train_df['clean_title']
            res_df['label']=train_df['2_way_label']
            res_df['id']=train_df['id']
        elif i=='test.pkl':
            res_df['text']=test_pubblic_df['clean_title']
            res_df['label']=test_pubblic_df['2_way_label']
            res_df['id']=test_pubblic_df['id']
        elif i=='valid.pkl':
            res_df['text']=validate_df['clean_title']
            res_df['label']=validate_df['2_way_label']
            res_df['id']=validate_df['id']
        res_pkl[i]=res_df
else:
    res_pkl=load_pkl(base_dir)

clip,clip_processor,clip_tokenizer=load_clip_model()
if TEXT_EMD_FLAG:
    for k, v in res_pkl.items():
        text_emds = []
        for i in tqdm(range(len(v))):
            text = v.loc[i, 'text']
            text_emd = clip_textual_feature_extraction(clip_tokenizer, clip, text)
            text_emds.append(text_emd)
        v['textual_feature_embedding'] = text_emds
        print(k + ':textual_feature_embedding done')
    if SAVE_FLAG:
        save_pkl(res_pkl,base_dir)
if VISUAL_EMD_FLAG:

    for k, v in res_pkl.items():
        visual_emds = []
        for i in tqdm(range(len(v))):
            image_path = os.path.join(images_dir, str(v.loc[i, 'id']) + '.jpg')
            try:
                visual_emd = clip_visual_feature_extraction(clip_processor, clip, [image_path])
            except:
                print('error:', image_path)
                visual_emd = None
            visual_emds.append(visual_emd)
        v['visual_feature_embedding'] = visual_emds
        print(k + ':visual_feature_embedding done')
    if SAVE_FLAG:
        save_pkl(res_pkl, base_dir)
if RETRIEVAL_FLAG:
    database_df=pd.concat([res_pkl['train.pkl'],res_pkl['valid.pkl']],axis=0)
    database_textual_tensor = torch.tensor(np.array(database_df['textual_feature_embedding'].tolist())).cuda()
    database_retrieval_tensor=torch.tensor(np.array(database_df['retrieval_embedding'].tolist())).cuda()
    database_visual_tensor = torch.tensor(np.array(database_df['visual_feature_embedding'].tolist())).cuda()
    database_label_tensor=torch.tensor(database_df['label'].tolist()).cuda()
    id_list = database_df['id'].tolist()
    database_df.reset_index(drop=True,inplace=True)
    for k,v in res_pkl.items():
        retrieved_item_id_list = []
        retrieved_item_similarity_list = []
        retrieved_indices_list=[]
        retrieve_label_list = []

        for i in tqdm(range(len(v))):
            v_vec=v['retrieval_embedding'][i]
            similarity_list_top_k,indices_list_top_k  = compute_similarity_and_take_top_k(v_vec, database_retrieval_tensor, 10)
            id_list_top_k = [id_list[j] for j in indices_list_top_k]
            retrieved_item_id_list.append(id_list_top_k)
            retrieved_indices_list.append(indices_list_top_k)
            retrieved_item_similarity_list.append(similarity_list_top_k)
            retrieve_label_list.append(database_label_tensor[indices_list_top_k].cpu().tolist())

        v['retrieved_item_id_list']=retrieved_item_id_list
        v['retrieved_item_similarity_list']=retrieved_item_similarity_list
        v['retrieved_indices_list']=retrieved_indices_list
        v['retrieved_label_list']=retrieve_label_list
        print(k+':retrieved_item_id_list done')

    if SAVE_FLAG:
        save_pkl(res_pkl,base_dir)
        with open(base_dir+'database_textual_tensor'+'.pt', 'wb') as f:
            torch.save(database_textual_tensor, f)
        with open(base_dir+'database_visual_tensor'+'.pt', 'wb') as f:
            torch.save(database_visual_tensor, f)
print('done')

