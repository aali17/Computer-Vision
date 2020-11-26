import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from build_model import model_tools

model=model_tools()
model_folder='checkpoints'
#image=sys.argv[1]
#img=cv2.imread(image)
#img=cv2.imread('raw_traffic_light/traffic_light_images/test/green/00febbe1-a9ae-4b5f-b682-8ebfdae485a3.jpg')
#img=cv2.imread('test_set/cats/cat.4001.jpg')
#img=cv2.imread('test_set/Boat/img_0.png')
img1=io.imread('test/yellow/7.jpg')
#img=cv2.imread('rawdata/superman/superman_00b89e7d-b40c-47ea-be96-8e9c0b7dcaf7.png')
session=tf.Session()
img=cv2.resize(img1,(100,100))
img=img.reshape(1,100,100,3)
labels = np.zeros((1, 3))

#Create a saver object to load the model
#saver = tf.train.import_meta_graph(os.path.join(model_folder,'.meta'))
saver = tf.train.import_meta_graph("trained/trained_variables.ckpt.meta")

#restore the model from our checkpoints folder

#Uncomment the following line for running on a windows machine
#saver.restore(session,os.path.join(model_folder,'.\\'))
saver.restore(session,"trained/trained_variables.ckpt")

#The following line is for running on a linux machine, comment it out if running on a windows machine
#saver.restore(session,os.path.join(model_folder,'./'))

#Create graph object for getting the same network architecture
graph = tf.get_default_graph()

#Get the last layer of the network by it's name which includes all the previous layers too
network = graph.get_tensor_by_name("add_4:0")

#create placeholders to pass the image and get output labels
im_ph= graph.get_tensor_by_name("Placeholder:0")
label_ph = graph.get_tensor_by_name("Placeholder_1:0")

#Inorder to make the output to be either 0 or 1.
network=tf.nn.sigmoid(network)

# Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {im_ph: img, label_ph: labels}
result=session.run(network, feed_dict=feed_dict_testing)


if result[0][0] == result.max():
    print('green')
    fig, ax = plt.subplots(figsize= (8,8))
    ax.imshow(img1)
    ax.set_title('Green test image')
    ax.text(0.2, 0.9, 'Green: ' + str(result.max()*100) + str('%'))
    ax.axis('off')
    plt.savefig('tested_img/g18.jpg', dpi=None, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format=None,
    transparent=False, bbox_inches=None, pad_inches=0.1,
    frameon=None, metadata=None)

elif result[0][1] == result.max():
    print('red')
    fig, ax = plt.subplots(figsize= (8,8))
    ax.imshow(img1)
    ax.set_title('Red test image')
    ax.text(0.2, 0.9, 'Red: ' + str(result.max()*100) + str('%'))
    ax.axis('off')

    plt.savefig('tested_img/g11.jpg', dpi=None, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format=None,
    transparent=False, bbox_inches=None, pad_inches=0.1,
    frameon=None, metadata=None)
else:
    print('yellow')
    fig, ax = plt.subplots(figsize= (8,8))
    ax.imshow(img1)
    ax.set_title('Yellow test image')
    ax.text(0.2, 0.9, 'Yellow: ' + str(result.max()*100) + str('%'))
    ax.axis('off')

    plt.savefig('tested_img/g27.jpg', dpi=None, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format=None,
    transparent=False, bbox_inches=None, pad_inches=0.1,
    frameon=None, metadata=None)

