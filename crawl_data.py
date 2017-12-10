


import os
import requests
import json
import numpy as np

if os.path.exists('data'):
    data_path = 'data'
else:
    os.mkdir('data')
    data_path = 'data'


def crawl_train(store_path=None):
    ## download train qa
    path = "https://storage.googleapis.com/kaggle-competitions-data/kaggle/7834/train-v1.1.json?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1512651399&Signature=R0orPe9Cjpw2uav9KZqI2mCwJ1k%2BBYI4841dCgBxEF%2Fkwg%2FO6FuMa6Qy4BgKBoOTbL9elQkhByrWby9Kx%2FKhGr4Upu21BGnviY4%2F49mnNfOChLnuLNNwWXYb1yey5CrbEn9PmQArs4ayRiaWas36spXbKcMeB1%2Btt8yPx3WWZ6rQnab6iveTf9%2Bz2UgjhmlIZGd4hdmmJWEJUsIwqFyxT4t0cMMXiFYzJZr2p6q8dgv5BFTvg9YvouTYocWnDp6yFikTJqiueOHwKCdGZRFXzKYp0XNndSyOvpCRdulhPk5sYSWjzxwNnoCIAOBWN%2BpYVD5Vfw24%2F%2Brl8QMVAC5o%2FA%3D%3D"
    res = requests.get(path)
    train = json.loads(res.text)
    
    #set path to store data
    if store_path:
        data_path = store_path
    else:
        global data_path
        
    #save original data (it's dictionary)
    with open(os.path.join(data_path,"train.json"),"w") as f:
        json.dump(train,f)
        f.close()
        
    #produce question list with ( title , context_id , context , qa_id , qa , qa_answer_start , qa_answer_text  )
    question_list=[]
    qa_count=0
    para_id=0
    for wiki in train:
        title =wiki['title'] 
        for para in wiki['paragraphs']:
            para_id+=1
            context = para['context']  
            for qa in para['qas']:
                qa_count+=1
                QA=[title,para_id,context]
                QA.extend([qa['id'],qa['question'] ,qa['answers'][0]['answer_start'],qa['answers'][0]['text'] ])
                question_list.append(QA)
    print('Total number train of:\n','context : {}\n'.format(para_id),'QA      : {}\n'.format(qa_count))
    np.save(os.path.join(data_path,'train.npy') , question_list )

    
    
def crawl_test(store_path=None):
    ## download test qa
    path = "https://storage.googleapis.com/kaggle-competitions-data/kaggle/7834/test-v1.1.json?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1512652008&Signature=E8%2BisF%2Fy2g9L%2Fmd6VSfdNPlYxGK4%2Bil6NDsSXh%2FNxQdsX7QRM09qH%2F6D4RchsjDNz2SEGrXKhmCeNPISs2%2FSNV8YNC%2BaJ0Yjg0rpVTa4U5w6mamB9Dukr0wUG%2BMNA9p%2BeWnurVTGc0zwk28yuNOkhHXNLDnohmlb8IYSKQvjTZpu3IxW8cHgCuP40l03r%2F1Iqu0aiyWuO05pn73fQeVFBef9%2F2azRINQCP07hnPoaHKad87L8thvf9pyzZQeihYRC5cgUGwbsxnDuBPYIzOcnj1E6His%2FhEVRkVBB8rvo3G57iJgtehkp9y6w7eoYttGGuYjQaALrB0Bj2lmTLDoOA%3D%3D"
    res = requests.get(path)
    data = json.loads(res.text)
    
    #set path to store data
    if store_path:
        data_path = store_path
    else:
        global data_path
        
    #save original data (it's dictionary)
    with open(os.path.join(data_path,"test.json"),"w") as f:
        json.dump(data,f)
        f.close()
    
    #produce question list with ( title , context_id , context , qa_id , qa  )
    question_list=[]
    qa_count=0
    para_id=0
    for wiki in test:
        title =wiki['title'] 
        for para in wiki['paragraphs']:
            para_id+=1
            context = para['context']  
            for qa in para['qas']:
                qa_count+=1
                QA=[title,para_id,context]
                QA.extend([qa['id'],qa['question']])
                question_list.append(QA)
    print('Total number test of:\n','context : {}\n'.format(para_id),'QA      : {}\n'.format(qa_count))
    question_list = np.array(question_list)
    np.save(os.path.join(data_path,'test.npy') , question_list )   
    
    
    
    
if __name__='__main__':
    import sys
    try:
        p = sys.argv[1]
        crawl_train(path = p)
        crawl_test(path = p)
    except NameError:
        crawl_train()
        crawl_test()
    
    




