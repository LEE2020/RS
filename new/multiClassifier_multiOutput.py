# 单输入，多输出，多任务的网络
# input1  
# output 1 ，年龄回归预测 
# output2 , 收入多分类预测
# output3 ， 性别二分类预测

from  keras import layers 
from keras import Input 
from keras.models import Model

vocal_size = 50000
num_income_group = 10 

# input processing  
posts_input = Input(shape=(None,),dtype='int32',name='posts')
embedd_posts = layers.Embedding(256,vocal_size)(posts_input)
x = layers.Conv1D(128,5,activation='relu')(embedd_posts) 
x = layers.MaxPooling1D(5)(x) 
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256，5，activation='relu')(x)
x = layers.Conv1D(256,5,activation='relu')(x) 
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128,activation='relu')(x) 


# output1 
age_prediction = layers.Dense(1,name='age')(x) 

# output2
income_prediction = layers.Dense(num_income_group,activation='softmax',name='income')(x)

# output 3
gender_prediction = layers.Dense(1,activation='sigmoid',name='gender')(x) 

# model 
model = Model(posts_input,[age_prediction,income_prediction,gender_prediction]) 

# compile and loss and acc

model.compile(optimizer='rmsprop',\
              loss = ['mse','categorical_crossentropy','binary_crossentropy'],\
              loss_weight = [0.25,1,10]) #    平衡不同度量下的损失 
# or model.compile(optimizer='rmsprop', \
#             loss = {'age':'mse' , 'income':'categorical_crossentropy' ,'gender':'binary_crossentropy'}，\
#             loss_weight = {'age':0.25,  'income':1 , 'gender':10 })


 
