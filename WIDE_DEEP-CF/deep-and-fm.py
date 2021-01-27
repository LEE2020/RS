# wide and deep algorithm - https://arxiv.org/abs/1606.07792 Wide & Deep Learning for Recommender Systems
# reference1  data -  https://archive.ics.uci.edu/ml/datasets/adult 
# reference https://github.com/kaitolucifer/wide-and-deep-learning-keras/blob/master/wide_and_deep.py 
# reference https://github.com/jrzaurin/Wide-and-Deep-Keras/blob/master/wide_and_deep_keras.py
import numpy as np 
import pandas as pd
from  sklearn.preprocessing  import LabelEncoder,StandardScaler,PolynomialFeatures 
import keras 
from keras.layers.normalization import BatchNormalization

LABEL_COLUMNS = ['income_bracket']
COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
    'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 
    'hours_per_week', 'native_country','income_bracket']
CATEGORY_COLUMNS = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country']
CONTINUOUS_COLUMNS = ['age','education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

def preprocessing():
    train = pd.read_csv('./data/adult.data',names = COLUMNS)
    train.dropna(how='any',axis = 0 )
    test = pd.read_csv('./data/adult.test',skiprows = 1 ,names =COLUMNS)
    test.dropna(how='any',axis = 0 ) 
    all_data = pd.concat([train,test])
    all_data['label'] = all_data['income_bracket'].apply(lambda x: '>50K' in x ).astype(int)
    y = all_data['label'].values
    del all_data['income_bracket'] 
    del all_data['label']
    
    for col in CATEGORY_COLUMNS:
        all_data[col] = LabelEncoder().fit_transform(all_data[col])
   
    #for col in CONTINUOUS_COLUMNS:
    #    all_data[col] = pd.DataFrame(StandardScaler().fit_transform(all_data[col]))
    #print(all_data)
    data_category = pd.DataFrame(all_data,columns = CATEGORY_COLUMNS)
    data_continuous = pd.DataFrame(all_data,columns = CONTINUOUS_COLUMNS)


    train_size = len(train)
    train_data_category = np.array(data_category.iloc[:train_size]) 
    test_data_category = np.array(data_category.iloc[train_size:] )
    train_data_conti = np.array(data_continuous.iloc[train_size:])
    test_data_conti = np.array(data_continuous.iloc[train_size:])
    scale = StandardScaler()
    train_data_conti = scale.fit_transform(train_data_conti) # 按照train data 的scaler来处理test data 
    test_data_conti = scale.transform(test_data_conti)   

    train_label = y[:train_size]
    test_label = y[train_size:] 
    #t = ['age','hours_per_week']
    #print(scale.fit_transform(all_data[t]))
    return train_data_category,train_data_conti,test_data_category,test_data_conti,train_label,test_label,all_data 
    
class WIDE_AND_DEEP(object):
    def __init__(self,mode = 'wide_and_deep'):
        self.mode = mode 
        train_data_category,train_data_conti,test_data_category,test_data_conti,train_label,test_label,all_data = preprocessing()
        self.train_data_category = train_data_category
        self.train_data_conti = train_data_conti
        self.test_data_category = test_data_category 
        self.test_data_conti = test_data_conti 
        self.train_label = train_label
        self.test_label = test_label 
        self.all_data = all_data 
        # interaction interaction_only (a,b)=>(1,a,b,ab)  false (a,b) =>(1,a,b,a^2,b^2,ab)  
        self.poly = PolynomialFeatures(degree = 2, interaction_only = True)
        self.train_poly_category = self.poly.fit_transform(self.train_data_category)
        self.test_poly_category = self.poly.transform(self.test_data_category)
        self.model = None 
        self.logistic_input = None 
        
        self.conti_input = None
        self.category_input = None
        self.deep_outlayer = None 
        
    def wide(self):
        samples_size, dim  = self.train_poly_category.shape
        self.logistic_input = keras.layers.Input(shape=(dim,)) 
    def deep(self):
        category_input = []
        category_embedding = []
       
        for i in range(len(CATEGORY_COLUMNS)):
            input_i = keras.layers.Input(shape=(1,))
            
            dim = len(np.unique(self.all_data[CATEGORY_COLUMNS[i]]))
            embedding_dim = int(np.ceil(dim**0.25))
            embedding_i = keras.layers.Embedding(dim,embedding_dim,input_length = 1)(input_i)
            flatten_i = keras.layers.Flatten()(embedding_i)
            category_input.append(input_i)
            category_embedding.append(flatten_i) 
         
        conti_input = keras.layers.Input(shape = (len(CONTINUOUS_COLUMNS),))
        conti_dense = keras.layers.Dense(256)(conti_input)
        
        concat_embeds = keras.layers.concatenate([conti_input] + category_embedding) 
        concat_embeds = keras.layers.Activation('relu')(concat_embeds)
        bn_embeds = keras.layers.normalization.BatchNormalization()(concat_embeds)
        
        fc1 = keras.layers.Dense(512)(bn_embeds)
        ac1 = keras.layers.advanced_activations.RELU()(fc1)
        bn1 = keras.layers.normalization.BatchNormalization()(ac1)
        fc2 = keras.layers.Dense(512)(bn1)
        ac2 = keras.layers.advanced_activations.RELU()(fc2)
        bn2 = keras.layers.normalization.BatchNormalization()(ac2)
        fc3 = keras.layers.Dense(128)(bn2)
        ac3 = keras.advanced_activations.RELU()(fc3) 
    
        self.conti_input = conti_input 
        self.category_input = category_input 
        self.deep_outlayer = ac3 

    def train(self,epoch,optimizer = 'adam',batch_size = 128):
        if self.mode == 'wide_and_deep':
            input_data = [self.x_train_conti] +\
                         [self.x_train_categ[:, i] for i in range(self.x_train_categ.shape[1])] +\
                         [self.x_train_categ_poly]

        
         
if __name__ == "__main__":
    train_data_category,train_data_conti,test_data_category,test_data_conti,train_label,test_label,all_data =  preprocessing() 
    obj = WIDE_AND_DEEP()
    obj.deep()
