import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import to_categorical
from keras.utils import plot_model
from matplotlib import pyplot as pl
from datetime import datetime
from keras import regularizers



now = datetime.now()
time_in = now.strftime("%H:%M:%S")

IMAGES = 0
LABELS = 1
INPUT_SHAPE_X = 128
INPUT_SHAPE_Y = 128
OUTPUT_NUM = 20
EPOCH = 50
LR = 0.0001
BATCH_SIZE = 32
regu = 0.005

train_images = []
train_labels = []
test_images = []
test_labels = []


train_data = np.load("images_arrays/TRAIN128x128.npy", allow_pickle=True)
test_data = np.load("images_arrays/TEST128x128.npy", allow_pickle=True)

for i in range(0,10):
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

LENGHT_TRAIN = len(train_data)
LENGHT_TEST = len(test_data)


def get_model():


    model = models.Sequential()

    model.add(layers.Conv2D(8, (5, 5) ,input_shape=(INPUT_SHAPE_X, INPUT_SHAPE_Y, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (5, 5)))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(32, (5, 5)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (5, 5)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256 ,activation='relu',kernel_regularizer=regularizers.l2(regu)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(regu)))
    model.add(layers.Dense(OUTPUT_NUM, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(learning_rate=LR), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


for i in range(0, LENGHT_TRAIN):
    train_images.append(train_data[i][IMAGES])
    train_labels.append(train_data[i][LABELS])

train_images = np.array(train_images)
train_labels = np.array(train_labels)

for i in range(0, LENGHT_TEST):
    test_images.append(test_data[i][IMAGES])
    test_labels.append(test_data[i][LABELS])

test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images.reshape((LENGHT_TRAIN, INPUT_SHAPE_X , INPUT_SHAPE_Y,1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((LENGHT_TEST, INPUT_SHAPE_X , INPUT_SHAPE_Y,1))
test_images = test_images.astype('float32') / 255



model = get_model()


history = model.fit(x=train_images, y=train_labels, epochs=EPOCH, batch_size=BATCH_SIZE, shuffle=True,validation_split=0.15)


test_loss, test_acc = model.evaluate(test_images, test_labels)

print("LOSS=", test_loss, "\t", "ACC=%{}".format(test_acc * 100))

pre = model.predict([test_images])

test_images = test_images.astype('float32') * 255
test_images = test_images.astype('uint8')
test_images = test_images.reshape((LENGHT_TEST, INPUT_SHAPE_X, INPUT_SHAPE_Y))

#file = "LR-"+str(LR)+"-ACC-"+str(test_acc*100)+"-batch-"+str(BATCH_SIZE)+"-regu-"+str(regu)+"INP-"+str(INPUT_SHAPE_X)+".png"
#plot_model(model, to_file=file, show_shapes=True, expand_nested=True, dpi=420)

#model.save(file.replace('.png', '_MODEL'), overwrite=True)

now = datetime.now()
time_out = now.strftime("%H:%M:%S")

print("TIME IN:{}\tTIME OUT:{}".format(time_in, time_out))
