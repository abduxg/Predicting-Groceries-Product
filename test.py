from keras import models
from keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as pl

test_data = np.load("images_arrays/TEST128x128.npy", allow_pickle=True)

LENGHT_TEST = len(test_data)

test_images = []
test_labels = []
IMAGES = 0
LABELS = 1
INPUT_SHAPE_X = 128
INPUT_SHAPE_Y = 128

for i in range(0, LENGHT_TEST):
    test_images.append(test_data[i][IMAGES])
    test_labels.append(test_data[i][LABELS])

test_images = np.array(test_images)
test_labels = np.array(test_labels)

test_labels = to_categorical(test_labels)

test_images = test_images.reshape((LENGHT_TEST, INPUT_SHAPE_X, INPUT_SHAPE_Y, 1))
test_images = test_images.astype('float32') / 255

model = models.load_model("LR-0.0001-ACC-32.91925489902496-batch-32-regu-0.005INP-128_MODEL")

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("LOSS=", test_loss, "\t", "ACC=%{}".format(test_acc * 100))

pre = model.predict([test_images])

test_images = test_images.astype('float32') * 255
test_images = test_images.astype('uint8')
test_images = test_images.reshape((LENGHT_TEST, INPUT_SHAPE_X, INPUT_SHAPE_Y))

true_number = 0
for i in range(0, LENGHT_TEST):
    a = np.argmax(pre[i])
    if test_labels[i][a] == 1.0:
        true_number += 1

print("TEST picture: {} Known picture number:{}".format(LENGHT_TEST, true_number))

category = []
with open("category","r") as file:
    for i in file.readlines():
        category.append(i)


while (True):
    print("For exit 'exit'")
    inp = input("Enter index on test images 0-{}:".format(LENGHT_TEST))
    if inp== "exit":
        break
    try:
        cat = np.argmax(pre[int(inp)])
        print(category[cat])
        pl.imshow(test_images[int(inp)])
        pl.show()
    except:
        print("Try again")
