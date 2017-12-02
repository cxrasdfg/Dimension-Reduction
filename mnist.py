import numpy as np
import struct

def load_mnist_training(num):
    print("training data loading")
    original_data_file = open('mnist/train-images',"rb")
    label_file = open('mnist/train-labels',"rb")
    data_buff = original_data_file.read()
    label_buff = label_file.read()

    data_index = 0
    label_index = 0

    # read the magic, image numbers, rows, columns
    magicD, numImages , numRows , numColumns = struct.unpack_from('>IIII' , data_buff , data_index)
    data_index += struct.calcsize('>IIII')

    magicL, numLabels = struct.unpack_from('>II' , label_buff , label_index)    
    label_index += struct.calcsize('>II') 

    data = np.zeros((num,28*28))
    labels = []

    for i in range(num):
        # read data
        im = struct.unpack_from('>784B',data_buff, data_index)
        data_index += struct.calcsize('>784B')
        im = np.mat(im)
        data[i]=im

        #read labels
        la = struct.unpack_from('>1B',label_buff, label_index)   
        label = la[0] 
        label_index += struct.calcsize('>1B')  
        labels.append(label) 
        
    print("training data loaded")
    return data, labels

def load_mnist_testing(num):
    print("testing data loading")
    original_data_file = open('mnist/t10k-images',"rb")
    label_file = open('mnist/t10k-labels',"rb")
    data_buff = original_data_file.read()
    label_buff = label_file.read()

    data_index = 0
    label_index = 0

    # read the magic, image numbers, rows, columns
    magicD, numImages , numRows , numColumns = struct.unpack_from('>IIII' , data_buff , data_index)
    data_index += struct.calcsize('>IIII')

    magicL, numLabels = struct.unpack_from('>II' , label_buff , label_index)    
    label_index += struct.calcsize('>II') 


    testdata = np.zeros((num,28*28))
    testlabels = []

    for i in range(num):
        # read data
        im = struct.unpack_from('>784B',data_buff, data_index)
        data_index += struct.calcsize('>784B')
        im = np.mat(im)
        testdata[i]=im

        #read labels
        la = struct.unpack_from('>1B',label_buff, label_index)   
        label = la[0] 
        label_index += struct.calcsize('>1B')  
        testlabels.append(label) 
    print("testing data loaded")
    return testdata, testlabels