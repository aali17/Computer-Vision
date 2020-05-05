import numpy as np 
import cv2
import os
from call_network import call_net 

im_ph, label_ph, network, session = call_net()

path, dirs, files_red = next(os.walk('raw_traffic_light/traffic_light_images/test/red'))
red_images = len(files_red)

path, dirs, files_yellow = next(os.walk('raw_traffic_light/traffic_light_images/test/yellow'))
yellow_images = len(files_yellow)

path, dirs, files_green = next(os.walk('raw_traffic_light/traffic_light_images/test/green'))
green_images = len(files_green)

image_numbers_by_color = [red_images, yellow_images, green_images]

folder_name = ['red', 'yellow', 'green']

correct_label = 0
missclass = 0
red_as_green = 0

for j in range(len(image_numbers_by_color)):
    individual_images = image_numbers_by_color[j]
    folder = folder_name[j]
    for i in range(individual_images):
        im=cv2.imread('raw_traffic_light/traffic_light_images/test/' + str(folder) + '/' + str(i) + '.jpg')
                       
#        img=cv2.imread('./raw_traffic_light/traffic_light_images/test/red/0.jpg')
        
        im=cv2.resize(im,(100,100))
        img=im.reshape(1,100,100,3)
        
        labels = np.zeros((1, 3))
        
        # Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {im_ph: img, label_ph: labels}
        result=session.run(network, feed_dict=feed_dict_testing)
        
        if folder == 'red' and result[0][1] == result.max():
            correct_label += 1
        elif folder == 'green' and result[0][0] == result.max():
            correct_label += 1
        elif folder == 'yellow' and result[0][2] == result.max():
            correct_label += 1
        else:
            missclass += 1
            # write all missclassified images in a folder to see what they look
            cv2.imwrite('raw_traffic_light/traffic_light_images/test/misclassified/' + str(folder) + str(i) + '.jpg', im)
        
        # we never want a red signal classified as green. want to see how many red 
        # signals are classified as green
        if folder == 'red' and result[0][0] == result.max():
            red_as_green += 1

accuracy = correct_label/(correct_label+missclass)
print(accuracy)
print('number of missclassified images: ' + str(missclass) + ' out of ' + str(correct_label+missclass))
print('number of red signals classified as green: ' + str(red_as_green))


