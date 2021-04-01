from keras import Input,layers

input_tensor = Input(shape=(32,))
dense = layers.Dense(32,activation='relu') #一个层是一个函数
output_tensor=dense(input_tensor)

# 对应sequential()模型，API的实现方法
from keras.models import Sequential,Model 
from keras import layers 
from keras import Input 

#before 
model = Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(64,)))
model.add(layers.Dense(32,activation='relu'))
model.add(lauers.Dense(10,activation='softmax'))

# now 
input_tensor = Input(shape=(64,))
x = layers.Dense(32,activation='relu')(input_tensor)
x = layers.Dense(32,activation='relu')(x)
output_tensor = layers.Dense(10,activation='softmax')(x) 

model = Model(input_tensor,output_tensor) # 模型将输入张量和输出张量转换为一个模型 

# model parameter 
model.summary()
