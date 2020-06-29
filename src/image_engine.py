import cv2
import numpy as np

#resmi gridlere bölerek ortalamasını alıp 64x64 boyutuna düşürüyor
def engine64x(path):
    img = cv2.imread(path, 0)
    # GETTING IMAGE NP ARRAY
    img256x = cv2.resize(img, (256, 256))
    soup = []
    arr_img256x = np.ndarray.tolist(img256x)

    a = 0
    b = 4
    c = 0

    for i in range(0, 64):
        while c < 256:
            soup.append(arr_img256x[c][a:b])
            c += 1
        if b == 256:
            break
        a += 4
        b += 4
        c = 0

    array = np.asarray(soup)
    array = np.reshape(array, (4096, 4, 4))

    imgarr_64x = []

    for i in array:
        imgarr_64x.append(int(np.mean(i)))

    imgarr_64x = np.array(imgarr_64x)
    imgarr_64x = np.reshape(imgarr_64x, (64, 64))

    imgarr_64x = imgarr_64x.astype("uint8")

    # Transpoz because picture rotating
    return imgarr_64x.T

#resmi gridlere bölerek ortalamasını alıp 128x128 boyutuna düşürüyor
def engine128x(path):
    img = cv2.imread(path, 0)
    # GETTING IMAGE NP ARRAY
    img256x = cv2.resize(img, (256, 256))
    soup = []
    arr_img256x = np.ndarray.tolist(img256x)

    a = 0
    b = 2
    c = 0

    for i in range(0, 128):
        while c < 256:
            soup.append(arr_img256x[c][a:b])
            c += 1
        if b == 256:
            break
        a += 2
        b += 2
        c = 0

    array = np.asarray(soup)
    array = np.reshape(array, (16384, 2, 2))

    imgarr_128x = []

    for i in array:
        imgarr_128x.append(int(np.mean(i)))

    imgarr_128x = np.array(imgarr_128x)
    imgarr_128x = np.reshape(imgarr_128x, (128, 128))

    imgarr_128x = imgarr_128x.astype("uint8")
    #Transpoz because picture rotating
    return imgarr_128x.T
