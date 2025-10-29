from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import  fashion_mnist,cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam   
from keras import regularizers

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


#to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#architecture
model=Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(1024,activation="relu"))
model.add(Dense(512,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(10,activation="softmax"))


#compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training
history=model.fit(x_train,y_train,
          epochs=10,
          batch_size=128,
          validation_split=0.2)
              

#evaluate
loss,test_accuracy=model.evaluate(x_test,y_test)   
print(f'test accuracy:{test_accuracy}') 

#visualize training history
import matplotlib.pyplot as plt
plt.plot(history.history['val_accuracy'],label='without regularization',color='blue')
plt.plot(history.history['val_accuracy'],label='without regularization',color='green')

plt.title('Model validation accuracy')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


#MODEL 2 :l2 regularizer (le-4) and dropout

model_1e_2=Sequential()
model_1e_2.add(Flatten(input_shape=(32,32,3)))
model_1e_2.add(Dense(1024,activation="relu",kernel_regularizer=regularizers.l2(1e-2)))
model_1e_2.add(Dense(512,activation="relu",kernel_regularizer=regularizers.l2(1e-2)))
model_1e_2.add(Dense(256,activation="relu",kernel_regularizer=regularizers.l2(1e-2)))
model_1e_2.add(Dense(128,activation="relu",kernel_regularizer=regularizers.l2(1e-2)))
model_1e_2.add(Dense(64,activation="relu",kernel_regularizer=regularizers.l2(1e-2)))
model_1e_2.add(Dense(10,activation="softmax",kernel_regularizer=regularizers.l2(1e-2)))

#compile
model_1e_2.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training
history_MODEL_LE2=model_1e_2.fit(x_train,y_train,
          epochs=10,
          batch_size=128,
          validation_split=0.2)
              
#evaluate
loss,test_accuracy=model_1e_2.evaluate(x_test,y_test)
print(f'test accuracy:{test_accuracy}')



