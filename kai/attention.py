


import os
import re

import numpy as np

from numpy import array
from keras.models import Sequential , Model
from keras.layers import Dense ,Input 
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import merge


from keras.layers import Multiply
from keras.layers import Lambda, Input
#from keras.backend import softmax #無法設定axis
from keras.activations import softmax #可設定axis

length=2
my_input = Input(shape=(length,5))
ls = LSTM(10,return_sequences=True)(my_input)
## output is h

## u = w*h , use linear trainsform . It can use another function like tanh ..etc
td = TimeDistributed(Dense(1,activation='linear',use_bias=False))(ls)

## alpha = softmax(u)
sm = Lambda(lambda x : softmax(x,axis=1) , name='softmax')(td)

## c = alpha*h
attn = Multiply()([sm,ls])

out = LSTM(5,return_sequences=True)(attn)
model = Model(inputs=[my_input] , outputs=[out])
model.compile(loss = 'mse',optimizer='adam')
model.summary()

model_1 = Model(inputs=[my_input] , outputs=[ls])
model_2 = Model(inputs=[my_input] , outputs=[sm])
model_3 = Model(inputs=[my_input] , outputs=[attn])


x = np.arange(10).reshape((-1,length,5))
y = x.reshape((-1))[::-1].reshape((-1,length,5))
#model.fit(x,y,epochs=1000,verbose=0)
model.predict(x,batch_size=1)


a1 = model_1.predict(x)
a2 = model_2.predict(x)
a3 = model_3.predict(x)
(a1[0][1]*a2[0][1]==a3[0][1]).all()
