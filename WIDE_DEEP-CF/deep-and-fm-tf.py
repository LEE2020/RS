# base on tensorflow 
# reference https://github.com/ChenglongChen/tensorflow-DeepFM/blob/master/DeepFM.py
# [1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
#      Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.


import tensorflow as tf 
import numpy as np 
import pandas as pd 
from  wide-and-deep-keras import  preprocessing as pp 

LABEL_COLUMNS = ['income_bracket']
COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
      'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss',
      'hours_per_week', 'natfive_country','income_bracket']
CATEGORY_COLUMNS = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country']
CONTINUOUS_COLUMNS = ['age','education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
s_model_name ='deep_and_fm'
wf = open(s_model_name + ".log", "w+")

def preprocessing()
    ''' genereate data from wide-and-deep-keras.pp() 
        array:
        train_data_category 
        train_data_conti
        test_data_category 
        test_data_conti
        
        label :
        train_label
        test_label

        dataframe:
        all_data 
    '''
    train_data_category,train_data_conti,test_data_category,test_data_conti,train_label,test_label,all_data = pp() 
    return  train_data_category,train_data_conti,test_data_category,test_data_conti,train_label,test_label,all_data 

class DEEP_AND_FM():
   ''' deep and fm class'''
    def __init__(self,layers = [32,32,32 ] , learning_rate = 0.001,batch_size = 256,epochs = 10 ,use_fm = True,use_deep = True \
                embedding_size = 8 , optimizer = 'adam', feature_size,field_size, self.loss_type = 'logloss' ):
        self.feature_size = feature_size   # sample_size 
        self.field_size = field_size       # features dim 
        self.learning_rate = learning_rate 
        self.batch_size = batch_size
        self.epochs = epochs 
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.embedding_size = embedding_size 
        self.layers = layers 
        self.loss_type = loss_type
        self.optimizer = optimizer 
       
        
        # init graph 
        self.graph = tf.Graph()
        with self.graph.as_default(): # 当有一个主线程的时候可以不写，直接调用tf.get_default_graph() 
            self.feats_index = tf.placeholder(tf.int32, shape=[None,None] , name = "feats_index")
            self.feats_value  = tf.placeholder(tf.float32,shape = [None,None],name = "feats_value")
            self.feats_label = tf.placeholder(tf.float32,shape=[None,1] , name = "label")
            self.weights = init_weights() 
            
            # model 
            self.embedding = tf.nn.embedding_lookup(self.weights['feature_embedding'], self.feats_index)
            feats_value = tf.reshape(self.feats_value , shape = [-1, self.field_size,1])
            self.embedding = tf.multiply(self.embbeind, feats_value) # element wise ,x,y the same vetor 
            
            # fm first order 
            self.first_order = tf.nn.embedding_lookup(self.weights['feature_embedding'],self.feats_index)
            self.first_order = tf.reduce_sum(tf.multiply(self.first_order,feats_value),axis = 2)
            self.first_order = tf.nn.dropout(self.frist_order, keep_prob = 0.8) 
            # fm second order 
            self.second_order_1 = tf.reduce_sum(self.embedding, axis = 1)
            self.second_order_1 = tf.square(self.second_order_1)
            self.second_order_2 = tf.square(self.embedding)
            self.second_order_2 = tf.reduce_sum(self.second_order_2,axis =1) 
            
            self.second_order = 0.5 * tf.subtract(self.second_order_1,self.second_order_2)
            self.second_order = tf.nn.dropout(self.second_order,keep_prob= 0.8) 
            # deep component
            self.deep = tf.reshape(self.embedding,shape = [-1,self.field_size * self.embedding_size])
            self.deep = tf.nn.dropout(self.deep, keep_prob = 0.8)
            for ind in range(len(self.layers)):
                self.deep = tf.add(tf.matmul(self.deep,self.weights['layer_%d' %ind] ), self.weights['bias_%d' %ind])
                self.deep = tf.nn.relu(self.deep)
                self.deep = tf.nn.dropdout(self.deep,keep_prob = 0.8)


            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.first_order,self.second_order,self.deep],axis = 1)
            elif self.use_fm and not self.use_deep:
                concat_input = tf.concat([self.first_order,self.second_order],axis = 1 )
            elif not self.use_fm and self.use_deep:
                concat_input = self.deep

            self.out = tf.add(tf.matmul(concat_input,self.weights['concat_layer']),self.weights['concat_bias'])
            self.out = tf.nn.softmax(self.out, name = 'prob') 
            # loss
            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmod(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label,self.out))

            # optimizer
            if  self.optimizer  == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate , beta1 = 0.9 ,beta2 = 0.999, espilon = 1e-8).minimize(self.loss)
            elif self.optimizer = 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.learning_rate,initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer = 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            elif self.optiizer = 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate,momentum = 0.95).minimize(self.loss)
            # run 
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)


    def init_weights():
        weights = dict()
        weights['feature_embedding'] = tf.Variable(tf.random_normal([self.feature_size,self.embedding_size],  \
                                        mean = 0.0 ,stddev = 1 ), name = "feature_embedding") 
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size,1], \
                                        mean = 0.0, stddev = 1) , name = "feature_bias" )
         
        nums_layer = len(self.layers)
        input_size = self.field_size * self.embedding_size 
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0])) # 缩放因子
        weights['layer_0'] = tf.Variable(np.random.normal(loc = 0.0 , scale = glorot , size = (input_size,self.layers[0])),dtype = np.float32)
        weights['bias_0'] = tf.Variable(np.random.normal(loc = 0.0 ,scale = glorot, size = (1, self.layers[0])), dtype = np.float32)           
        # init other layers 
        for ind in range(1,nums_layer):
            glorot = np.sqrt(2.0 / (self.layers[ind-1] + self.deep_layers[ind])) # 缩放因子 
            weights['layer_%d' %ind] = tf.Variable(np.random.normal(loc = 0.0 ,scale = glorot,size = ( self.layers[ind-1] , self.layers[ind])),\
                                       dtype = np.float32)
            weights['bias_%d'] %ind] = tf.Variable(np.random.normal(loc = 0.0 ,scale = glorot,size = (self.layers[ind-1], self.layers[ind])), \
                                        dtype =np.float32) 
        # the final layer

        if self.use_fm and not self.use_deep:
            input_size = self.field_size + self.embedding_size 
        elif not self.use_fm and self.use_deep:
            input_size = self.layers[-1]
        else:
            input_size = self.field_size + self.embedding_size + self.layers[-1] 
        glorot = np.sqrt(2.0 / (input_size + 1)) 
        
        weights['concat_layer'] = tf.Variable(np.random.normal(loc = 0.0, scale = glorot, size = (input_size,1)),dtype =np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype = np.float32)
    
        return weights 


    def calc_ks(y_true, y_prob, n_bins=10):
        percentile = np.linspace(0, 100, n_bins + 1).tolist()
        bins = [np.percentile(y_prob, i) for i in percentile]
        bins[0] = bins[0] - 0.01
        bins[-1] = bins[-1] + 0.01
        binids = np.digitize(y_prob, bins) - 1
        y_1 = sum(y_true ==1)
        y_0 = sum(y_true ==0)
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))
        bin_false = bin_total - bin_true
        true_pdf = bin_true / y_1
        false_pdf = bin_false / y_0
        true_cdf = np.cumsum(true_pdf)
        false_cdf = np.cumsum(false_pdf)
        ks_list = np.abs(true_cdf - false_cdf).tolist()
        ks = max(ks_list)
        return ks

    def train(self,train_data,train_data_index,valid_data_index,valid_data,train_label,valid_label,early_stopping=True):
        ''' train '''
        shuffle_indices = np.random.permutation(np.arange(len(train_data)))
        for epoch in self.epochs:
            train_prob_list = [] 
            total_batch = len(train_label)/self.batch_size 
            for ind in range(total_batch):
                start = ind * self.batch_size 
                end = (ind+1)* self.batch_size 
                end = end if end < len(train_label) else len(train_label)
                train_batch_index = train_data_index[shuffle_indices[start:end],:]
                train_batch_data = train_data[shuffle_indices[start:end],]
                train_batch_label = train_label[shuffle_indices[start:end],:]
                feed_dict = {self.feat_index: train_batch_index, self.feat_value : train_batch_data , self.label : train_batch_label }
                tf.sess.run(optimizer,feed_dict = feed_dict) # 希望从网络中得到的数据，填写在 run（）中 ,训练 optimizer  
        # train_loss, train_ks etc. 
        block = 10000
        for j in range(len(train_label)/block):        
            start = j*block
            end = (j+1)*block if (j+1)*block < len(train_label) else len(train_label)    # run(loss,prob)
            train_loss, prob = self.sess.run([self.loss,self.prob ], \
                                feed_dict = {self.feat_index: train_data_index[start:end],\
                                self.feat_value : train_data[start:end] , \
                                self.label : train_label[start:end] })
                
            for i in range(end-start):
                train_prob_list.append(prob[i,0])
            train_ks = cal_ks(train_label,np.narray(train_prob_list)) 
            wf.write(" block: %d , train dataset :  loss=%f ,  ks=%f  " % (j,  train_loss, train_ks))   
                
        eval_prob_list = [] 
        total_batch = len(valid_label)/ block 
        for j  in range(total_batch):
            start = ind * block
            end = (ind+1)* block
            end = end if end < len(valid_label) else len(valid_label) # run(loss,prob)
            valid_loss, prob = self.sess.run([self.loss,self.prob ], \
                                 feed_dict = {self.feat_index: valid_data_index[start:end],\
                                 self.feat_value : valid_data[start:end] , \
                                 self.label : valid_label[start:end] })
            for  i in range(end-start):
                eval_prob_list.append(prob[i,0])
            valid_ks = cal_ks(valid_ks,np.narray(eval_prob_list))
            wf.write(" block: %d , train dataset :  loss=%f ,  ks=%f  " % (j,  train_loss, train_ks)) 
                                     
