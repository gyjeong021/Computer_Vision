import numpy as np
import cv2 as cv

def embossing(img):
    femboss = np.array([[-1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0]])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray16 = np.int16(gray)
    emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss) + 128, 0, 255))

    return emboss