# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 19:49:26 2017

@author: Siddharth.C
"""

import numpy as np
import pandas as pd
import re
import gc
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import roc_auc_score as auc

#read the data
train_data = pd.read_json('../input/train_data.json',orient="index")
test_data = pd.read_json('../input/test_data.json',orient='index')

train_data.reset_index(level = 0, inplace = True)
train_data.rename(columns={'index':'ID'}, inplace=True)

test_data.reset_index(level = 0, inplace = True)
test_data.rename(columns={'index':'ID'}, inplace=True)

train_data = train_data.replace({'segment':{'pos':1,'neg':0}})
train_data['segment'].value_counts()/train_data.shape[0]

Y=train_data.segment.values
tr_id=train_data.ID.values
te_id=test_data.ID.values
train_data.drop(['ID','segment'], axis=1, inplace=True)
test_data.drop(['ID'], axis=1, inplace=True)
full_data=train_data.append(test_data)

#Some cleanup
full_data['genres']=[re.sub(pattern='IndiaVsSa',repl='Cricket',string=x) for x in full_data['genres']]
full_data['genres']=[re.sub(pattern='Formula1',repl='FormulaOne',string=x) for x in full_data['genres']]
full_data['genres']=[re.sub(pattern='Table Tennis',repl='TableTennis',string=x) for x in full_data['genres']]

full_data['cities']=[re.sub(pattern='Bengaluru', repl='Bangalore',string=x) for x in full_data['cities']]
full_data['cities']=[re.sub(pattern=' ', repl='_',string=x) for x in full_data['cities']]
full_data['titles']=[re.sub(pattern=' ', repl='_',string=x) for x in full_data['titles']]

#get all genres watched by each user (as list)
full_data['g1'] = [re.sub(pattern='\:\d+',repl='',string=x) for x in full_data['genres']]
full_data['g1'] = full_data['g1'].apply(lambda x: x.split(','))

#similarly for dow, tod, cities, titles
full_data['g2'] = [re.sub(pattern='\:\d+', repl='', string = x) for x in full_data['dow']]
full_data['g2'] = full_data['g2'].apply(lambda x: x.split(','))

full_data['g3'] = [re.sub(pattern='\:\d+', repl='', string = x) for x in full_data['tod']]
full_data['g3'] = full_data['g3'].apply(lambda x: x.split(','))

full_data['g4'] = [re.sub(pattern='\:\d+', repl='', string = x) for x in full_data['cities']]
full_data['g4'] = full_data['g4'].apply(lambda x: x.split(','))

full_data['g5'] = [re.sub(pattern='\:\d+', repl='', string = x) for x in full_data['titles']]
full_data['g5'] = full_data['g5'].apply(lambda x: x.split(','))

#get the time spent watching each genre
full_data['g1_1'] = [re.sub(pattern='[a-zA-Z]+:', repl='',string=x) for x in full_data['genres']]
full_data['g1_1'] = full_data['g1_1'].apply(lambda x: x.split(','))

#similarly for for columns
full_data['g2_1'] = [re.sub(pattern='[0-9]+:', repl='',string=x) for x in full_data['dow']]
full_data['g2_1'] = full_data['g2_1'].apply(lambda x: x.split(','))

full_data['g3_1'] = [re.sub(pattern='[0-9]+:', repl='',string=x) for x in full_data['tod']]
full_data['g3_1'] = full_data['g3_1'].apply(lambda x: x.split(','))

full_data['g4_1'] = [re.sub(pattern='[A-Za-z0-9_]+:', repl='',string=x) for x in full_data['cities']]
full_data['g4_1'] = full_data['g4_1'].apply(lambda x: x.split(','))


full_data=full_data.reset_index(drop=True)

#fuction to count number of unique values
def get_counts(x, dict1):
    for row in x:
        for e in row:
            if e in dict1:
                dict1[e]+=1
            else:
                dict1[e]=1
    return dict1
                    
all_genres=dict()
all_genres=get_counts(full_data.g1, all_genres)

all_dow=dict()
all_dow=get_counts(full_data.g2, all_dow)

all_tod=dict()
all_tod=get_counts(full_data.g3, all_tod)

all_cities=dict()
all_cities=get_counts(full_data.g4, all_cities)

all_titles=dict()
all_titles=get_counts(full_data.g5, all_titles)

all_cities=pd.DataFrame.from_dict(all_cities, "index")
all_cities=all_cities.reset_index()
all_cities.columns=['city','counts']
all_cities.sort_values('counts', ascending=False, inplace=True)
top_cities=all_cities.city[:100] #top 100 cities

all_titles=pd.DataFrame.from_dict(all_titles, "index")
all_titles=all_titles.reset_index()
all_titles.columns=['title','counts']
all_titles.sort_values('counts', ascending=False, inplace=True)
top_titles=all_titles.title[:300] #top 300 most viewed titles

#split up the lists earlier created, 1 column for each genre (or tod/dow)
#column value = time spent watching
#[There might be more elegant ways to do this, but this works well enough]
def convert_wide (x, col,col_1, names):
    for n in names:
        col_name=str(col)+"_"+str(col)+"_"+str(n)
        l1=[]
        i=0
        for row in x[col]:
            j=0
            for e in row:
                if e==n:
                    l1.append(int(x[col_1][i][j]))
                    break
                j+=1
            if(len(l1) ==i):
                l1.append(0)
            i+=1
        l1=pd.Series(l1)
        x[col_name] = l1
        del l1
        
    return x

full_data=convert_wide(full_data.copy(), "g1", "g1_1", all_genres)
full_data=convert_wide(full_data.copy(), "g2", "g2_1", all_dow)
full_data=convert_wide(full_data.copy(), "g3", "g3_1", all_tod)

sum_night=full_data.iloc[:,55:61].sum(axis=1) #total time spent watching between 8 pm & 2 pm
sum_allhrs=full_data.iloc[:,55:].sum(axis=1) #totaltime spent all hours inclusive

sum_weekends=full_data.iloc[:,53:55].sum(axis=1) #time spent watching on weekends
sum_alldays=full_data.iloc[:,48:55].sum(axis=1) #all days

full_data['ratio_night']=sum_night.astype('float') / sum_allhrs
full_data['ratio_weekends']=sum_weekends.astype('float')/ sum_alldays

#top 6 most popular genres are cricket, comedy, drama, romance, reality & talkshow
#compute respective ratios
sum_allgenres=full_data.iloc[:,14:48].sum(axis=1)
full_data['ratio_cricket'] = full_data.g1_g1_Cricket.astype('float') /sum_allgenres #ratio for cricket genre
full_data['ratio_comedy'] = full_data.g1_g1_Comedy.astype('float') /sum_allgenres
full_data['ratio_drama'] = full_data.g1_g1_Drama.astype('float') /sum_allgenres
full_data['ratio_romance'] = full_data.g1_g1_Romance.astype('float') /sum_allgenres
full_data['ratio_reality'] = full_data.g1_g1_Reality.astype('float') /sum_allgenres
full_data['ratio_talkshow'] = full_data.g1_g1_TalkShow.astype('float') /sum_allgenres

#OHE for top 100 cities & top 300 titles. 
#(Just 1/0. Not bothered about time)
#[Agian, there might be more elegant ways to do this.]
def convert_ohe (x, col, names):
    for n in names:
        col_name=str(col)+"_"+str(n)
        l1=[]
        i=0
        for row in x[col]:
            j=0
            for e in row:
                if e==n:
                    l1.append(1)
                    break
                j+=1
            if(len(l1) ==i):
                l1.append(0)
            i+=1
        l1=pd.Series(l1)
        x[col_name] = l1
        del l1
        
    return x

full_data=convert_ohe(full_data.copy(), "g4", top_cities.values)  
full_data=convert_ohe(full_data.copy(), "g5", top_titles.values)  

del sum_alldays, sum_allgenres, sum_allhrs, sum_night, sum_weekends

#This stuff is pulled up from Manish Saraswat's benchmark script.
#like cleaning up blank values, total count of genres, titles, etc.
w1 = full_data['titles']
w1 = w1.str.split(',')

main = []
for i in np.arange(full_data.shape[0]):
    d1 = w1[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main.append(nest)
    
blanks = []
for i in np.arange(len(main)):
    if '' in main[i]:
        print "{} blanks found".format(len(blanks))
        blanks.append(i)
        
#replacing blanks with 0
for i in blanks:
    main[i] = [x.replace('','0') for x in main[i]]
    
#converting string to integers
main = [[int(y) for y in x] for x in main]

#adding the watch time
tosum = []
for i in np.arange(len(main)):
    s = sum(main[i])
    tosum.append(s)
    
full_data['title_sum'] = tosum
        
def wcount(p):
    return p.count(',')+1
    
full_data['title_count'] = full_data['titles'].map(wcount)
full_data['genres_count'] = full_data['genres'].map(wcount)
full_data['cities_count'] = full_data['cities'].map(wcount)
full_data['tod_count'] = full_data['tod'].map(wcount)
full_data['dow_count'] = full_data['dow'].map(wcount)

full_data.drop(['cities','dow','genres','titles','tod','g1','g2','g3',
                'g1_1','g2_1','g3_1','g4','g4_1', 'g5'], inplace=True, axis=1)
              
train_data=full_data[:len(train_data)]
test_data=full_data[len(train_data):]

#There was an id leak in the roadsigns problem. Let's just add an id feature here also just in case
#I used the number in the id string rather than row number itself.
#[doesn't cause much difference to be honest]
tr_id1=pd.Series([re.sub(pattern='train-',repl='',string=x) for x in tr_id]).astype('float')/200000
te_id1=pd.Series([re.sub(pattern='test-',repl='',string=x) for x in te_id]).astype('float')/100000


train_data['id1']=tr_id1
test_data['id1']=te_id1

###LightGBM model###
lgb_params = {
    'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True,
    'num_leaves': 2**6 -1, 'learning_rate': 0.01, 'max_depth': 6,
    'max_bin': 2**6 -1, 'metric': 'auc',
    'colsample_bytree': 0.12, #0.2 
    'reg_alpha': 0.02, 'reg_lambda': 0.02,
    'min_child_weight': 10, 'min_child_samples': 10, 
    'bagging_fraction': 0.9, 
    'bagging_freq': 10}
    
nrounds = 3000  
kfolds = 5  
oof_train=pd.DataFrame({'ID': tr_id, 'segment': 0})
best=[]
score=[]
#5-fold CV. Save OOF preds
skf = SKF( n_splits=kfolds, shuffle=True,random_state=123)
i=0
for train_index, test_index in skf.split(train_data, Y):
    print('Fold {0}'.format(i + 1))
    X_train, X_val = train_data.iloc[train_index], train_data.iloc[test_index]
    y_train, y_val = Y[train_index],Y[test_index]

    ltrain = lgb.Dataset(X_train,y_train)
    lval = lgb.Dataset(X_val,y_val, reference= ltrain)

    gbdt = lgb.train(lgb_params, ltrain, nrounds, valid_sets=lval,
                         verbose_eval=50,
                         early_stopping_rounds=50)  
    bst=gbdt.best_iteration
    pred=gbdt.predict(X_val, num_iteration=bst)
    oof_train.loc[test_index,"segment"]= pred
    scr=auc(y_val,pred) 
    
    best.append(bst)    
    score.append(scr)
    i+=1
    
    del ltrain
    del lval
    del gbdt
    gc.collect()

print(np.mean(score))
print(np.mean(best))

#oof_train.to_csv('../output/oof/lgb3_oof_tr.csv', index=False)

#retrain on full train set
best_nrounds=int(round(np.mean(best)))
ltrain=lgb.Dataset(train_data,Y)

gbdt = lgb.train(lgb_params, ltrain, best_nrounds,verbose_eval=50)
preds=gbdt.predict(test_data)

#save & submit
df_out=pd.DataFrame({'ID': te_id, 'segment': preds})
df_out.to_csv('../output/lgb_hotstar_submission.csv', index=False)
