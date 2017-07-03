# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 11:40:53 2017

@author: Siddharth.C
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import log_loss as ll
import gc

df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')
sample_sub=pd.read_csv('../input/sample_submission.csv')

df_train.columns=['Id','DetectedCamera','AngleOfSign','SignAspectRatio','SignWidth','SignHeight','Target']

train_id=df_train.Id.values
test_id=df_test.Id.values
Y=df_train.Target

df_train.drop(['Id','Target'], axis=1, inplace=True)
df_test.drop(['Id'], axis=1, inplace=True)

#There's clearly a leak in this dataset. The oder of rows is a significant feature in determining the target.
#This could be because of a temporal nature of data collection, & random split of train & test.
#hence consecutive rows could be from the same car along the same driving route.

#Row Id percentile feature
df_train['Id_Perc']=pd.Series(range(len(df_train))).astype('float') / len(df_train)
df_test['Id_Perc']=pd.Series(range(len(df_test))).astype('float') / len(df_test)

df_full=df_train.append(df_test)

df_full.DetectedCamera.replace({'Front':0,'Left':1, 'Rear':2, 'Right':3}, inplace=True)
Y.replace({'Front':0,'Left':1, 'Rear':2, 'Right':3}, inplace=True)
Y=Y.values

#convert angle to radians. 
#compute features like Sin, Cos, Tan, & it's product with other given features
df_full.AngleOfSign=np.radians(df_full.AngleOfSign)
df_full['Sin']=np.sin(df_full.AngleOfSign) 
df_full['Cos']=np.cos(df_full.AngleOfSign) 
df_full['SignArea']=df_full.SignWidth * df_full.SignHeight 

df_full['MirrorAngle']=max(df_full.AngleOfSign)-df_full.AngleOfSign #360 - Angle
df_full['sqrtAngle']=df_full.AngleOfSign ** 0.5
df_full['Angle_Ht']=df_full.AngleOfSign * df_full.SignHeight
df_full['Angle_AR']=df_full.AngleOfSign * df_full.SignAspectRatio
df_full['Tan']=df_full.Sin / df_full.Cos
df_full['SinCos']=df_full.Sin * df_full.Cos

df_full=pd.get_dummies(df_full, columns=['DetectedCamera'], sparse=False) #OHE

#Split back to train & test
df_train=df_full[:len(df_train)]
df_test=df_full[len(df_train):]

###XGB###
dtest=xgb.DMatrix(df_test)
xgb_params = {
    'seed': 619, 
    'colsample_bytree': 0.67,
    'silent': 1,
    'subsample': 0.9,
    'learning_rate': 0.05,
    'objective': 'multi:softprob',
    'num_class': 4,
    'max_depth': 4, 
    'min_child_weight': 3, 
    'alpha': 0.02,
    'eval_metric' : 'mlogloss'
    
}

nrounds = 2000  
kfolds = 5  

#dataframe for oof predictions. 
#Useful to find ideal number of rounds, & optimizing weights while ensembling.
oof_train=pd.DataFrame({'ID': train_id, 'Front':0, 'Left':0, 'Rear':0, 'Right':0})

best=[]
score=[]

skf = SKF( n_splits=kfolds, shuffle=True,random_state=123)
i=0
for train_index, test_index in skf.split(df_train, Y):
    print('Fold {0}'.format(i + 1))
    X_train, X_val = df_train.iloc[train_index], df_train.iloc[test_index]
    y_train, y_val = Y[train_index],Y[test_index]

    dtrain = xgb.DMatrix(X_train,y_train)
    dval = xgb.DMatrix(X_val,y_val)
    watchlist = [(dtrain, 'train'), (dval, 'eval')]

    gbdt = xgb.train(xgb_params, dtrain, nrounds, watchlist,
                         verbose_eval=50,
                         early_stopping_rounds=25)  
    bst=gbdt.best_ntree_limit
    pred=gbdt.predict(dval, ntree_limit=bst)
    oof_train.loc[test_index,"Front"]= pred[:,0]
    oof_train.loc[test_index,"Left"]= pred[:,1]
    oof_train.loc[test_index,"Rear"]= pred[:,2]
    oof_train.loc[test_index,"Right"]= pred[:,3]
    scr=ll(y_val,pred) 
    
    best.append(bst)    
    score.append(scr)
    i+=1
    
    del dtrain
    del dval
    del gbdt
    gc.collect()

print(np.mean(score))
print(np.mean(best))

#Save oof preds
#oof_train.to_csv('../output/oof/xgb5_oof_tr.csv', index=False)

#retrain on whole train set for best nrounds
best_nrounds=int(round(np.mean(best)))
dtrain=xgb.DMatrix(df_train,Y)

watchlist = [(dtrain, 'train')]
gbdt = xgb.train(xgb_params, dtrain, best_nrounds,watchlist,verbose_eval=50,early_stopping_rounds=25)
pred=gbdt.predict(dtest)
pred=pd.DataFrame(pred, columns=['Front','Left','Rear','Right'])
pred['Id']=test_id
pred=pred[['Id','Front','Left','Rear','Right']]
pred.to_csv('../output/xgb_submission.csv', index=False)

##LightGBM###
lgb_params = {
    'boosting_type': 'gbdt', 'objective': 'multiclass',
    'num_class':4, 'nthread': -1, 'silent': True,
    'num_leaves': 2**4 -1, 'learning_rate': 0.05, 'max_depth': 4,
    'max_bin': 2**4 -1, 'metric': 'multi_logloss',
    'colsample_bytree': 0.6, 
    #'reg_alpha': 0.01, 'reg_lambda': 0.01,
    'min_child_weight': 5, 'min_child_samples': 10, 
    'bagging_fraction': 0.9, 
    'bagging_freq': 10}

nrounds = 2000  
kfolds = 5  
oof_train=pd.DataFrame({'ID': train_id, 'Front':0, 'Left':0, 'Rear':0, 'Right':0})
best=[]
score=[]

skf = SKF( n_splits=kfolds, shuffle=True,random_state=123)
i=0
for train_index, test_index in skf.split(df_train, Y):
    print('Fold {0}'.format(i + 1))
    X_train, X_val = df_train.iloc[train_index], df_train.iloc[test_index]
    y_train, y_val = Y[train_index],Y[test_index]

    ltrain = lgb.Dataset(X_train,y_train)
    lval = lgb.Dataset(X_val,y_val, reference= ltrain)

    gbdt = lgb.train(lgb_params, ltrain, nrounds, valid_sets=lval,
                         verbose_eval=50,
                         early_stopping_rounds=25)  
    bst=gbdt.best_iteration
    pred=gbdt.predict(X_val, num_iteration=bst)
    oof_train.loc[test_index,"Front"]= pred[:,0]
    oof_train.loc[test_index,"Left"]= pred[:,1]
    oof_train.loc[test_index,"Rear"]= pred[:,2]
    oof_train.loc[test_index,"Right"]= pred[:,3]
    scr=ll(y_val,pred) 
    
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

best_nrounds=int(round(np.mean(best)))
ltrain=lgb.Dataset(df_train,Y)

gbdt = lgb.train(lgb_params, ltrain, best_nrounds)
pred=gbdt.predict(df_test)
pred=pd.DataFrame(pred, columns=['Front','Left','Rear','Right'])
pred['Id']=test_id
pred=pred[['Id','Front','Left','Rear','Right']]
pred.to_csv('../output/lgb_submission.csv', index=False)


#Weighted ensemble. Gives marginal improvement
A=pd.read_csv('../output/xgb_submission.csv')
B=pd.read_csv('../output/lgb_submission.csv')

submit=A.copy()
for i in range(1,5):
    submit.ix[:,i] = A.ix[:,i] * 0.59 + B.ix[:,i] * 0.41

submit.to_csv('../output/ensemble_submission.csv', index=False)
