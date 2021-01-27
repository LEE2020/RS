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
        https://zhuanlan.zhihu.com/p/113350563 reference  ''' 
         
                
        
    def rf_lr(self,data,label,epoch==10): 
        ''' rf ,gbdt benchmark https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2264-5 ''' 
        

if __name__ == "__main__":
    job = RECO()
    data = job.generate_samples()
    data,label = job.generate_data(data)
    #job.lr_model(data,label)
    job.gbdt(data,label)

'''

## 建模
# 下面训练三个模型对数据进行预测， 分别是LR模型， GBDT模型和两者的组合模型， 然后分别观察它们的预测效果， 对于不同的模型， 特征会有不同的处理方式如下：
# 1. 逻辑回归模型： 连续特征要归一化处理， 离散特征需要one-hot处理
# 2. GBDT模型： 树模型连续特征不需要归一化处理， 但是离散特征需要one-hot处理
# 3. LR+GBDT模型： 由于LR使用的特征是GBDT的输出， 原数据依然是GBDT进行处理交叉， 所以只需要离散特征one-hot处理
    # 把训练集和测试集分开
    # 模型预测
    y_pred = lr.predict_proba(test)[:, 1]  # predict_proba 返回n行k列的矩阵，第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率, 这里的1表示点击的概率
    print('predict: ', y_pred[:10]) # 这里看前10个， 预测为点击的概率




### LR + GBDT建模
          
    
    model = gbm.booster_

    gbdt_feats_train = model.predict(train, pred_leaf = True)
    gbdt_feats_test = model.predict(test, pred_leaf = True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns = gbdt_feats_name) 
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns = gbdt_feats_name)

    train = pd.concat([train, df_train_gbdt_feats], axis = 1)
    test = pd.concat([test, df_test_gbdt_feats], axis = 1)
    train_len = train.shape[0]
    data = pd.concat([train, test])
    del train
    del test
    gc.collect()

    # # 连续特征归一化
    scaler = MinMaxScaler()
    for col in continuous_feature:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    for col in gbdt_feats_name:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)

    train = data[: train_len]
    test = data[train_len:]
    del data
    gc.collect()

    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.3, random_state = 2018)

    
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print('tr-logloss: ', tr_logloss)
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('val-logloss: ', val_logloss)
    y_pred = lr.predict_proba(test)[:, 1]
    print(y_pred[:10])


# 训练和预测lr模型
lr_model(data.copy(), category_fea, continuous_fea)

# 模型训练和预测GBDT模型
gbdt_model(data.copy(), category_fea, continuous_fea)

# 训练和预测GBDT+LR模型
gbdt_lr_model(data.copy(), category_fea, continuous_fea)
'''
