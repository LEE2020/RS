import tensorflow as tf 
from keras.datasets import imdb
# load data
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
# network 

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid')) 


# optimizor & loss 
model.compile(optimizer = 'rmsprop', \
              loss = 'binary_crossentropy',\
               metrics =['accuracy'])  # 监控了acc

# train model

model_rst = model.fit(train_data,train_labels,epochs=20,\
                    batch_size = 512,validation_data=(test_data,test_labels)) 

# parameters 
model_rst.history.keys()  # 里面包含监控的指标和loss（train & val) 
# plot loss 
import matplotlib.pyplot as plt 
history_dict = model_rst.history
loss_val = history_dict['loss']
val_loss_val = history_dict['val_loss']
epochs = range(1,len(loss_val)+1)

plt.plot(epochs,loss_val) 
plt.plot(epochs,val_loss_val)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show() 


# plot accuracy 

plt.clf() 
acc = history_dict['acc']
val_acc = history_dict['val_acc']




