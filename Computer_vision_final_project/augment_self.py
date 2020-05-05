import cv2
import os
import numpy as np

def augment_self(image_directory):
    
    path, dirs, files = next(os.walk(image_directory))
    images = len(files)
    
    for i in range(images):
        image = cv2.imread(image_directory + '/' + str(i) + '.jpg')
        rand1 = np.random.randint(2,7,1)[0]
        rand2 = np.random.randint(2,7,1)[0]
        cropped = image[rand1:99-rand1, rand2:99-rand2]
        cropped = cv2.resize(cropped,(100,100))
        cv2.imwrite(image_directory + '/' + str(i+images) + '.jpg', cropped)
        
    return 0


def call_augment(times_augmented, folder_link):
    for i in range(times_augmented):
        augment_self(folder_link)
    return 0

folder_link = 'data/yellow'
call_augment(1, folder_link)