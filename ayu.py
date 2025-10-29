from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar100
from keras.optimizers import Adam
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encode
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model 1
model_base = Sequential()
model_base.add(Flatten(input_shape=(32, 32, 3)))
model_base.add(Dense(1024, activation='relu'))
model_base.add(Dense(512, activation='relu'))
model_base.add(Dense(256, activation='relu'))
model_base.add(Dense(128, activation='relu'))
model_base.add(Dense(64, activation='relu'))
model_base.add(Dense(100, activation='softmax'))

# Compile
model_base.compile(optimizer=Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Train
model_base.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate
loss, test_accuracy = model_base.evaluate(x_test, y_test)
print(f"Base Model Accuracy: {test_accuracy:.4f}")

# Model 2
model_le4 = Sequential()
model_le4.add(Flatten(input_shape=(32, 32, 3)))
model_le4.add(Dense(1024, activation='relu'))
model_le4.add(Dense(512, activation='relu'))
model_le4.add(Dense(256, activation='relu'))
model_le4.add(Dense(128, activation='relu'))
model_le4.add(Dense(64, activation='relu'))
model_le4.add(Dense(100, activation='softmax'))

# Compile
model_le4.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train
model_le4.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate
loss, test_accuracy = model_le4.evaluate(x_test, y_test)
print(f"Model 2 Accuracy: {test_accuracy:.4f}")
