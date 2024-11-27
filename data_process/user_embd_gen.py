import os
import pandas as pd
from tqdm import tqdm
USE_CLIP=True
LOC=''
from transformers import AutoProcessor, CLIPModel, AutoTokenizer, CLIPTextModel, CLIPTextConfig,CLIPConfig,CLIPVisionConfig
def load_clip_model():
    model = CLIPModel.from_pretrained(LOC + 'PRETRAINED_MODEL/' + "clip-vit-base-patch32")

    processor = AutoProcessor.from_pretrained(LOC + 'PRETRAINED_MODEL/' + "clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained(LOC + 'PRETRAINED_MODEL/' + "clip-vit-base-patch32")
    return model, processor,tokenizer
def clip_textual_feature_extraction(tokenizer, model, text):
    if len(text) > 512:
        embd_list = []
        for i in range(0, len(text), 64):
            inputs = tokenizer(text[i:i + 64], padding=True, return_tensors="pt", max_length=77, truncation=True)

            emd = model.get_text_features(**inputs).cpu().detach().numpy()
            embd_list.append(emd)
        return np.mean(np.concatenate(embd_list), axis=0)
    else:
        inputs = tokenizer(text, padding=True, return_tensors="pt", max_length=77, truncation=True)

        emd = model.get_text_features(**inputs).cpu().detach().numpy()
        emd = np.mean(emd, axis=0)
        return emd




import numpy as np


validate_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/multimodal_only_samples/multimodal_valid.tsv')
test_pubblic_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/multimodal_only_samples/multimodal_test_public.tsv')
train_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/multimodal_only_samples/multimodal_train.tsv')
comments_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/all_comments.tsv')
images_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/public_image_set/')

pool_mode='mean'

train_pkl_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/train.pkl')
valid_pkl_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/valid.pkl')
test_pkl_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/test.pkl')

validate_df=pd.read_csv(validate_df_dir, sep='\t')
test_pubblic_df=pd.read_csv(test_pubblic_df_dir, sep='\t')
train_df=pd.read_csv(train_df_dir, sep='\t')
comments_df=pd.read_csv(comments_df_dir, sep='\t')
comments_df=comments_df.dropna(subset=['author','body'])
comments_df=comments_df.reset_index(drop=True)
from collections import defaultdict
user_embd_dict = defaultdict(list)
for row in tqdm(comments_df.itertuples(), total=len(comments_df)):
    user_embd_dict[row.author].append(row.body)
del comments_df
for df in [validate_df,test_pubblic_df,train_df]:
    df = df.dropna(subset=['author', 'clean_title'])
    df=df.reset_index(drop=True)
    for i in tqdm(range(len(df))):
        author=df.loc[i,'author']
        clean_title=df.loc[i,'clean_title']
        if author in user_embd_dict:
            user_embd_dict[author].append(clean_title)
        else:
            user_embd_dict[author]=[clean_title]
del validate_df
del test_pubblic_df
del train_df

user_embd_pd=pd.DataFrame()
user_embd_author_list=[]
user_embd_embd_list=[]
clip, clip_processor, clip_tokenizer = load_clip_model()

for k,v in tqdm(user_embd_dict.items()):
    user_embd_author_list.append(k)
    user_embd_embd_list.append(clip_textual_feature_extraction(clip_tokenizer,clip,v))
user_embd_pd['user_id']=user_embd_author_list
user_embd_pd['user_embedding']=user_embd_embd_list
user_embd_pd.to_pickle(os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/user_mean.pkl'))

