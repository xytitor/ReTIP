import os

from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
LOC=''
base_dir = LOC + 'DATA_SET/mm_fnd/fakeddit/'
text_emd_flag=False
visual_emd_flag=False
save_flag=True
load_flag=True
retreival_flag=True
def loading_model():
    local_path = LOC + 'PRETRAINED_MODEL/' + "blip-image-captioning-large"

    processor = BlipProcessor.from_pretrained(local_path)

    model = BlipForConditionalGeneration.from_pretrained(local_path).to("cuda")

    return processor, model
def convert_image_to_text(processor, model, image_path):
    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)

    text = processor.decode(out[0], skip_special_tokens=True)

    return text


blip=loading_model()

validate_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/valid.pkl')
test_pubblic_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/test.pkl')
train_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/train.pkl')
images_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/public_image_set/')

proceess_data_list=['valid.pkl']

res_pkl={}
for k in proceess_data_list:
    res_pkl[k]=pd.read_pickle(base_dir+k)
    caption_list=[]
    for row in tqdm(res_pkl[k].itertuples(), total=len(res_pkl[k])):
        image_path = os.path.join(images_dir, row.id + '.jpg')
        caption=convert_image_to_text(blip[0],blip[1],image_path)
        caption_list.append(caption)
    res_pkl[k]['caption']=caption_list
    res_pkl[k].to_pickle(base_dir+k)


