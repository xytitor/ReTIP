import os
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel, AutoTokenizer, CLIPTextModel, CLIPTextConfig,CLIPConfig,CLIPVisionConfig
import pandas as pd
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
LOC=''
USE_CLIP=True
base_dir = LOC + 'DATA_SET/mm_fnd/fakeddit/'



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

validate_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/valid.pkl')
test_pubblic_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/test.pkl')
train_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/train.pkl')
images_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/public_image_set/')
proceess_data_list=['valid.pkl','train.pkl','test.pkl']

res_pkl={}

clip, clip_processor, clip_tokenizer = load_clip_model()

for k in proceess_data_list:
    res_pkl[k]=pd.read_pickle(base_dir+k)
    retrieval_list=[]
    for row in tqdm(res_pkl[k].itertuples(), total=len(res_pkl[k])):
        text=row.text+','+row.caption
        retrieval_emd=clip_textual_feature_extraction(clip_tokenizer,clip,text)
        retrieval_list.append(retrieval_emd)
    res_pkl[k]['retrieval_embedding']=retrieval_list
    res_pkl[k].to_pickle(base_dir+k)
    del res_pkl[k]


