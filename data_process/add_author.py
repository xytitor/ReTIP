import os
import pandas as pd
from tqdm import tqdm
LOC=os.getenv()
validate_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/multimodal_only_samples/multimodal_valid.tsv')
test_pubblic_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/multimodal_only_samples/multimodal_test_public.tsv')
train_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/multimodal_only_samples/multimodal_train.tsv')
comments_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/all_comments.tsv')
images_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/public_image_set/')
user_df_dir=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/user_mean.pkl')
proceess_data_list=['test.pkl','train.pkl','valid.pkl']
user_df=pd.read_pickle(user_df_dir)
for i in proceess_data_list:
    print(i)
    author_list=[]
    user_embedding_list=[]
    loc=os.path.join(LOC, 'DATA_SET/mm_fnd/fakeddit/' + i)
    if i=='test.pkl':
        csv=pd.read_csv(test_pubblic_df_dir, sep='\t')
    elif i=='train.pkl':
        csv=pd.read_csv(train_df_dir, sep='\t')
    elif i=='valid.pkl':
        csv=pd.read_csv(validate_df_dir, sep='\t')
    pkl=pd.read_pickle(loc)
    user_mean_embedding = user_df['user_embedding']

    for row in tqdm(pkl.itertuples(), total=len(pkl)):
        author=csv.loc[csv['id']==row.id,'author'].values[0]
        author_list.append(author)
        try:
            user_embedding=user_df.loc[user_df['user_id']==author,'user_embedding'].values[0]
        except:
            user_embedding=user_mean_embedding
        user_embedding_list.append(user_embedding)
    pkl['user_embedding']=user_embedding_list
    pkl['author']=author_list
    pkl.to_pickle(loc)
