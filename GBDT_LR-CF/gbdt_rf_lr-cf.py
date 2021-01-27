#ata ../RS-CF/ml-1m/  评分数据转化为点击率的数据，认为评分高的电影点击率更高，更应该被推荐。
# 评分中>=3分，label = 1 ; 其他label = 0 
# 同一个用户对多部电影有过评分，可以形成多个sample ，每一个sample 都有对应的label。形成一条完整的样本。

## 数据导入与简单处理
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import gc
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')


"""数据读取与预处理"""

tags_file ='../RS-CF/ml-1m/users.dat'
ratings_file = '../RS-CF/ml-1m/ratings.dat'
movies_file ='../RS-CF/ml-1m/movies.dat'


class  RECO:
    '''  base on gbdt ,gbdt+lr, rf , rf +lr ,recommend system '''
    def generate_samples(self):
        ''' generate samples
            input :  users , movies ,rating
            output : userid , movies features , label ( if  rating >=3 then 1 else 0 ) 
        '''   
        tags = pd.read_csv(tags_file, sep='::', header=None, names=['user_id', 'gender', 'age', 'occuputation','zipcode'])
        ratings = pd.read_csv(ratings_file, sep='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
        movies = pd.read_csv(movies_file, sep='::', header=None, names=['movie_id', 'title', 'genres']) 
        combined_df = ratings.join(movies, on=['movie_id'], rsuffix='_r').join(tags, on=['user_id'], rsuffix='_t')
        del combined_df['movie_id_r']; del combined_df['user_id_t'] ; del combined_df['movie_id'] 
        combined_df['label'] = combined_df['rating'].map(lambda x:(x>=3.0 and 1 ) or (x<3 and 0 ))
        del combined_df['rating']
        combined_df.dropna(axis=0, how='any', inplace=True)
        #combined_df = combined_df.pivot_table(columns=['label','age','gender','occuputation','zipcode','timestamp','genres'\
        #,'title'], index=['user_id'],values = ['label','age','gender','occuputation','zipcode','timestamp','genres','title']).fillna(0).values 
        return combined_df[:1000]

    def generate_data(self,data ):
        ''' lr model 
            continue feats : minmax scaler
            category feats : one hot
        '''
        continue_feats = ['age']
        category_feats = ['gender','title','genres','occuputation','zipcode']
        scaler = MinMaxScaler()
      
        for feat in continue_feats:
            data[feat] = scaler.fit_transform(data[feat].values.reshape(-1,1))
        for feat in category_feats:
            onehot = pd.get_dummies(data[feat],prefix='_c') 
            del data[feat]
            data = pd.concat([onehot,data],axis=1)
             
        pos  = data[data['label'] == 1].iloc[-1000:]
        N = len(pos)
        neg =  shuffle(data[data['label'] == 0]).iloc[-N*8:]
        # before model , type data : dataframe to narray 
        data = pd.concat([pos , neg])
        label = data['label'].values 
        del data['label']
        del data['user_id']
        data = data.values 
        return data,label 

    def lr_model(self,data,label,epoch=10):
        data_train,data_test,label_train,label_test = train_test_split(data,label,test_size = 0.33,random_state= 10) 
        lr = LogisticRegression() 
        lr.fit(data_train,label_train)
        train_logloss = log_loss(label_train,lr.predict_proba(data_train))
        test_logloss = log_loss(label_test,lr.predict_proba(data_test))
        print('train logloss : %2.2f  , test logloss  , %2.2f  ' %( train_logloss, test_logloss ))   
        

    def gbdt(self,data,label,epoch= 10):
        ''' gbdt sklearn''' 
        data_train,data_test,label_train,label_test = train_test_split(data,label,test_size = 0.33,random_state= 10)
        gbm = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary',subsample=0.8,
                              min_child_weight=0.5,colsample_bytree=0.7,num_leaves=100,
                              max_depth=3,learning_rate=0.01,n_estimators=50
                             )
        gbm.fit(data_train, label_train,
             eval_set=[(data_train, label_train), (data_test, label_test)],
             eval_names=['train', 'test'],
             eval_metric='binary_logloss'
            )
        train_logloss = log_loss(label_train,gbm.predict_proba(data_train))
        test_logloss = log_loss(label_test,gbm.predict_proba(data_test))
        print('train logloss : %2.2f  , test logloss  , %2.2f  ' %( train_logloss, test_logloss ))
        
        
    def gbdt_lr(self,data,label,epoch=10):
        '''
        https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf  gbdt + lr 在广告点击率预估场景的使用 
        https://zhuanlan.zhihu.com/p/113350563 reference  
        input : data (dataframe) columns = ['user_id','title,'age','gender']
        gbdt feats: gbdt_leaf_0 ~ gbdt_leaf_n 
        onehot feats =  gbdt_leaf_0 : [1,0,0,...,0] 
        new input  =  input + onehot feats 

        ''' 
        data_train,data_val,label_train,label_val = train_test_split(data,label,test_size = 0.33,random_state= 10)
        gbm = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary',subsample=0.8,
                                min_child_weight=0.5,colsample_bytree=0.7,num_leaves=100,
                                max_depth=3,learning_rate=0.01,n_estimators=50
                               )    
        gbm.fit(data_train, label_train,
               eval_set=[(data_train, label_train), (data_val, label_val)],
               eval_names=['train', 'test'],
               eval_metric='binary_logloss'
              )       
        model = gbm.booster_
        gbdt_feats_train = model.predict(data_train, pred_leaf = True)
        gbdt_feats_test = model.predict(data_val, pred_leaf = True)
        
        gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
        df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns = gbdt_feats_name) 
        df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns = gbdt_feats_name)
        # narray transto dataframe
        data_train = pd.DataFrame(data_train)
        data_val = pd.DataFrame(data_val)
        train = pd.concat([data_train, df_train_gbdt_feats], axis = 1)
        val = pd.concat([data_val, df_test_gbdt_feats], axis = 1)
        train_len = train.shape[0]
        data = pd.concat([train,val])
        
        for col in gbdt_feats_name:
            onehot_feats = pd.get_dummies(data[col], prefix = col)
            del data[col]
            data = pd.concat([data, onehot_feats], axis = 1)
       
        x_train, x_val, y_train, y_val = train_test_split(data, label, test_size = 0.2, random_state = 10)
         
        lr = LogisticRegression() 
        lr.fit(x_train, y_train)
        tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
        val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])

        # NE = (-1) / len(y_pred_test) * sum(((1+y_test)/2 * np.log(y_pred_test[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_test[:,1])))
        y_pred_train = lr.predict_proba(x_train)[:,1]
        y_pred_val = lr.predict_proba(x_val)[:,1]
        val_ne = (-1) / len(y_pred_val) * sum(((1+y_val)/2 * np.log(y_pred_val) +  (1-y_val)/2 * np.log(1 - y_pred_val)))
        tr_ne = (-1) / len(y_pred_train) * sum(((1+y_train)/2 * np.log(y_pred_train) +  (1-y_train)/2 * np.log(1 - y_pred_train)))
        print('tr-logloss: %2.2f , val logloss:%2.2f  ' %(tr_logloss, val_logloss))
        print('tr-ne: %2.2f , val ne:%2.2f  ' %(tr_ne, val_ne))


                

    def rf_lr(self,data,label,epoch=10): 
        ''' rf ,gbdt benchmark https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2264-5 ''' 
        pass     

if __name__ == "__main__":
    job = RECO()
    data = job.generate_samples()
    data,label = job.generate_data(data)
   # job.lr_model(data,label)
    job.gbdt_lr(data,label)
   # job.rf_lr(data,label)

