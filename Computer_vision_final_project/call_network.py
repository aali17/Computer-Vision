import tensorflow as tf
from build_model import model_tools

def call_net():
    model=model_tools()
    model_folder='checkpoints'
    session=tf.Session()
    
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
    
    return im_ph, label_ph, network, session