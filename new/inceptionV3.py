#   多个分支
from keras import layers 
# 1x1 
branch_a = layers.Conv2D(128,1,activation='relu',strides=2)(x) 

#1x1 and 3x3 
branch_b = layers.Conv2D(128,1,activation='relu')(x)
branch_b = layers.Conv2D(128,3,activation='relu',strides=2)(branch_b) 

# 3x3 and 3x3
branch_c = layers.AveragePooling2D(3,strides= 2)(x)
branch_c = layers.Conv2D(128,3,activation='relu')(branch_c) 

# 1x1 3x3 3x3 
branch_d = layers.Conv2D(128,1,activation='relu')(x)
branch_d = layers.Conv2D(128,3,activation='relu')(branch_d)  
branch_d = layers.Conv2D(128,3,activation='relu',strides=2 )(branch_d)

# all output 

output=layers.concatnate([branch_a,branch_b,branch_c,branch_d],axis=-1) 

