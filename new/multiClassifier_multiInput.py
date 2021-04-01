# 多输入数据，利用API实现方式。不同来源的数据，对应不同的处理方式。（多模态） 
# input1: 文本text
# input2: 问题 question
# output ：回答 answer 

from  keras.models import Model 
from keras import layers
from keras import Input 

text_size = 10000
question_size = 10000
answer_size = 500

text_input = Input(shape=(None,),dtype='int32',name='tx') # 可变长的整数序列 作为输入 1
embedd_text = layers.Embedding(text_size,64)(text_input) # 嵌入到64维中
encoded_text= layers.LSTM(32)(embedd_text) 

question_input = Input(shape=(None,),dtype='int32'，name='qs') # 可变长的整数序列作为输入2 
embedd_question = layers.Embedding(question_size,32)(question_input)
encoded_question = layers.LSTM(16)(embedd_question) 

# 多个输入串联起来
concatnated = layers.concatenate([encoded_text,encoded_question]],axis = -1) 

#output
answer = layers.Dense(answer_size,activation='softmax')(concatnated) 

# model and loss and optimizor 
model = Model([text_input,question_input],answer)  # 实例化时，指定输入输出 
model.compile(optimizer='rmsprop',\
              loss = 'categorical_crossentropy',\
              metrics=['acc'])
# train model 

model.fit([text_data,question_data],answer_data,epochs=10,batch_size=128)
# or  model.fit({'tx':text_data,'qs':question_data},answer_data,epochs=10,batch_size =128 ) 

