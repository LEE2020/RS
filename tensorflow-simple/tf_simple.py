# reference : https://github.com/kaitolucifer/character-level-rnn-tensorflow 
# sample tensorflow 
import tensorflow as tf
import numpy as np 
import pandas as pd 

batch_size = 256 # sequences per batch 
num_step = 100  # number of sequence steps per batch 
lstm_size = 256 # hidden layer size of lstm 
num_layers = 2  # number of hidden layers 
learning_rate = 0.001
keep_prob = 0.8 
epochs = 10 
save_every_n = 100 

def load_data(path):
    data = [] 
    seq_data = []
    target = [] 
    with open(path) as f:
        for line in f.readlines():
            lines = line.strip().split()
            data_tmp = [lines[0],lines[1],lines[2]]
            label_tmp = lines[4]
            [ seq_tmp.append(1) for _ in range(len(data_tmp))]  # 表示真实的数据存在，防止rnn 发生梯度消失
            data.append(data_tmp)          
            target.append(label_tmp)
            
    data = np.narray(data)
    target = np.narray(target)
    seq_data = np.narray(seq_data)
    return data,target,seq_data        


class sampleRNN:
    def __init__(self,num_classes,learning_rate = 0.001,batch_size = 256,\
                num_layers= 2, num_step = 100, keep_prob=0.8,epochs= 10 ,\
                lstm_size = 256):
        tf.reset_default_graph()
        # input  
        inputs = tf.placeholder(tf.float32, shape = [None,None] , name = 'inputs')
        target = tf.placeholder(tf.int32,shape = [None,1] , name = 'target')
        #keep_prob = tf.placeholder(tf.float32, name ='keep_prob')  
        keep_prob = tf.constant(0.8) 
        # layer
        lstm_cell = []  
        for _ in range(num_layers):
            lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob = keep_prob)
            lstm_cell.append(drop)  
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
        self.initial_state = cell.zero_states(batch_size,tf.float32)

        x_onehot = tf.one_hot(cell, num_classes)
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        
        # output layer 
        seq_output = tf.concat(output,axis =1)
        x = tf.reshape(seq_output,[-1,lstm_size])
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([lstm_size,num_classes],stddev = 0.1))
            softmax_b = tf.Variable(tf.zeros(num_classes))
        # output 
        logits = tf.matmul(x, softmax_w) + softmax_b
        self.out = tf.nn.softmax(logits, name='predictions')

        # loss
        y_onehot = tf.one_hot(target,num_classes)
        y_onehot = tf.reshape(y_onehot, shape = logits.get_shape())
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_onehot)
        self.loss = tf.reduce_mean(loss)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate , beta1 = 0.9 ,beta2 = 0.999, espilon = 1e-8).minimize(loss) 
       

if __name__ == '__main__':
    ''' load data
        train graph
        train model and save to ....pckt 
    '''
    train_data,train_target,train_index = load_data('./train.file')
    valid_data,valid_target,valid_index = loda_data('./valid.file')
    num_classes = 2 
    model = sampleRNN(2,learning_rate = learning_rate ,batch_size = batch_size,\
                  num_layers= num_layers, num_step = num_step, keep_prob=keep_prob ,epochs= epochs ,\
                  lstm_size = lstm_size)

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        count = 1 
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            new_state = sess.run(model.initial_state)
            loss = 0
            shuffle_indices = np.random.permutation(np.arange(train_data)) 
            for iter_ in range(len(train_target)/batch_size):
                count += 1
                train_data_batch  = train_data[shuffle_indices[iter_*batch_size:(iter_+1)*batch_size],:]
                train_target_batch = train_target[shuffle_indices[iter_*batch_size:(iter_+1)*batch_size],:]

                start = time.time()
                feed_dict = {model.inputs : train_data_batch,\
                             model.target: train_target_batch, \
                         model.initial_state: new_state 
                        }
                batch_loss, new_state, _ = sess.run([model.loss, 
                                                    model.final_state, 
                                                    model.optimizer], 
                                                    feed_dict=feed_dict)
                end = time.time()
                if count % 100 == 0:
                    print('epoch {}/{} '.format(epoch +1 , epochs),\
                          'iteration {} '.format(count ) ,\
                           'batch loss {:.4f}'.format(batch_loss),\
                           'time {:.4f} sec/batch '.format((end-start)))
                if count % save_every_n == 0:
                    saver.save(sess,"checkpoints/l{}s{}.ckpt".format(count,lstm_size))
        saver.save(sess, "checkpoints/l{}s{}.ckpt".format(count,lstm_size))

    def test():

        ''' restore checkpoint , sess  to predict prob on test samples ''' 
    
        test_data,test_target,test_index = loda_data('./test.file')   
        checkpoint = tf.train.latest_checkpoint('checkpoints') 
        model = sampleRNN(2, lstm_size=lstm_size)  
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.restore(sess,checkpoint)
            new_state = sess.run(model.initial_state)
            feed_dict = {model.inputs : test_data,\
                         model.target: test_target, \
                          model.initial_state: new_state
                         }  
            

            out , _  = sess.run([model.out,model.final_state],feed_dict = feed_dict) 
        return out 

 
