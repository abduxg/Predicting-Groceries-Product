import os
import numpy as np
from image_engine import engine64x, engine128x

# change actually path images

path_fotos = "/home/abdux/tmp/images/"

photos_cat = os.listdir(path_fotos)
file = open("category", "w")
a = 0
for i in photos_cat:
    stre = i + "\t" + str(a) + "\n"
    file.writelines(stre)
    a += 1
file.close()

imgsTR = []

imgsTST = []



print("Veriler alınıyor")
COUNTER = 0
for cat in photos_cat:
    path_cat = path_fotos + cat
    directory = os.listdir(path_cat)

    directory_size = len(directory)
    TRAIN_LENGHT = directory_size * 0.85
    TEST_LENGHT = directory_size - TRAIN_LENGHT
    COUNTER = 0
    np.random.shuffle(directory)

    catnum = photos_cat.index(cat)
    for i in directory:
        path = path_cat + "/" + i
        print(path)
        if path.endswith(".png"):
            if COUNTER <= TRAIN_LENGHT:
                # Burada engine 64x fonksiyonu kullanılırsa resimler 64x64 olarak kaydeder
                photo_tr = engine128x(path)
                imgsTR.append([photo_tr, catnum])

                COUNTER += 1
            else:
                photo_tst = engine128x(path)
                imgsTST.append([photo_tst, catnum])



np.save("TRAIN128x128", imgsTR)



np.save("TEST128x128",imgsTST)
