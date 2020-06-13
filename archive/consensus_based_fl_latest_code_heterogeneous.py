#!/gpfs_common/share03/hdai/richeng/Sim1/venv/bin/python3.6
import tensorflow as tf
from tensorflow import keras

from collections import defaultdict
import numpy as np
import pandas as pd
import random
import math
import time
from sklearn.decomposition import PCA
from itertools import islice 
import heapq
import sys
import matplotlib.pyplot as plt
from copy import copy
import networkx as nx



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
#from keras.models import Sequential
#from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

#from tensorflow.keras import datasets, layers, models

#StrNum = sys.argv[1]
# B = 0.005
momentum_SGD=0.95
NODES = 20
LOSS_OBJECT = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
tf.keras.backend.set_floatx('float64')
np.random.seed(101)
random.seed(101)

#############################################
#############################################
#############################################
# Load it once and then comment it 
b = np.random.randint(0,10,size=(NODES,NODES))
Adj_mat = (b + b.T)/2
Adj_mat[Adj_mat <6] = 0
Adj_mat[Adj_mat >= 6] = 1
for i in range(0,len(Adj_mat)):
    Adj_mat[i][i]=0

G=nx.from_numpy_matrix(Adj_mat)
#print(nx.is_connected(G))
while (nx.is_connected(G) !=True):
    b = np.random.randint(0,10,size=(NODES,NODES))
    Adj_mat = (b + b.T)/2
    Adj_mat[Adj_mat <6] = 0
    Adj_mat[Adj_mat >= 6] = 1
    G=nx.from_numpy_matrix(Adj_mat)
    
#print(nx.is_connected(G))    
for i in range(0,len(Adj_mat)):
    Adj_mat[i][i]=0
np.save('Adj_mat',Adj_mat)
###########################################
###########################################
###########################################

Adj_mat=np.load('Adj_mat.npy')
print(Adj_mat)

def AssignDatasets2(nodes):
    mnist = keras.datasets.mnist   ## load the data set
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data() ## load the training and testing set
    train_images, test_images = train_images/255.0, test_images/255.0

    
    train_image_samples, test_image_samples = len(train_images), len(test_images)
    num_sample_per_node=int(np.floor(train_image_samples/NODES))
    num_sample_per_node_test=int(np.floor(test_image_samples/NODES))
    print(num_sample_per_node)
    indices_chsoen_for_nodes=np.random.permutation(len(train_images));
    indices_chsoen_for_nodes_test=np.random.permutation(len(test_images));
    train_dataset = zip(train_images, train_labels)
    test_dataset = zip(test_images, test_labels)

    train_images_by_label = defaultdict(list)
    test_images_by_label = defaultdict(list)

    train_dataset_by_node = defaultdict(list)
    test_dataset_by_node =defaultdict(list)
    for node in range (NODES):
            images=train_images[indices_chsoen_for_nodes[(node)*num_sample_per_node :(node)*num_sample_per_node+num_sample_per_node]]
            labels=train_labels[indices_chsoen_for_nodes[(node)*num_sample_per_node: (node)*num_sample_per_node+num_sample_per_node]]
            if len(train_dataset_by_node[node]) == 0:
                    train_dataset_by_node[node].append(images)
                    train_dataset_by_node[node].append(labels)

                
    for node in range (NODES):
            images=test_images[indices_chsoen_for_nodes_test[(node)*num_sample_per_node_test: (node)*num_sample_per_node_test+num_sample_per_node_test]]
            labels=test_labels[indices_chsoen_for_nodes_test[(node)*num_sample_per_node_test: (node)*num_sample_per_node_test+num_sample_per_node_test]]
            if len(test_dataset_by_node[node]) == 0:
                    test_dataset_by_node[node].append(images)
                    test_dataset_by_node[node].append(labels)          


    return train_dataset_by_node, test_dataset_by_node


def CreateModel(data_shape):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=data_shape),
        keras.layers.Dense(128, activation='relu', dtype='float64'),
        keras.layers.Dense(10, activation='softmax', dtype='float64')
    ])

    return model

def Loss(model, x, y):
    y_ = model(x)
    return LOSS_OBJECT(y_true=y, y_pred=y_)

def SetOptimizer(lr):
#    optimizer = tf.keras.optimizers.SGD(learning_rate=lr,momentum=momentum_SGD,nesterov=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    return optimizer 

def Grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = Loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def NumOfGrads(grads):
    count = 0
    for grad in grads:
        current_count = grad.shape[0]
        for dimension in grad.shape[1:]:
            current_count *= dimension
        count += current_count
    return count



def ClipGradsL2(grads, C):
    clipped_grads, global_norm = tf.clip_by_global_norm(grads, C)
    if global_norm > C:
        print(global_norm)
        print('global_norm larger than C')
    return clipped_grads

def ClipGradsL1(grads, C):
    norm_l1 = 0
    for grad in grads:
        norm_l1 += tf.norm(grad, ord=1)        
    clipped_grads,_ = tf.clip_by_global_norm(grads,C,use_norm=norm_l1)  
    print(norm_l1)

    return clipped_grads

#def CollectGradsAdv(model, batch_size, datasets, C, delta, clipping):
#    all_grads = []
#    for node in range(len(datasets)):
#        data_size = len(datasets[node][0])
#        candidate_index = list(range(data_size))
#        L = batch_size
#        if data_size <= batch_size:
#            sampled_index = candidate_index
#            L = data_size
#        else:
#            sampled_index = np.random.choice(candidate_index, batch_size, replace=False)
#        
#        batched_grads = []
#        new_grads = []
#        if clipping:
#            for index in sampled_index:
#                image, label = np.asarray([datasets[node][0][index]]), np.asarray([datasets[node][1][index]])
#                _, grads = Grad(model, image, label)
#                if delta > 0:
#                    grads = ClipGradsL2(grads, C)
#                elif delta == 0:
#                    grads = ClipGradsL1(grads, C)
#                else:
#                    raise ValueError("delta is {}, it cannot be negative!".format(delta))
#                    
#                if not batched_grads:
#                    batched_grads = grads
#                else:
#                    for i, grad in enumerate(grads):
#                        batched_grads[i] = tf.math.add(batched_grads[i], grad)
#            new_grads = [grad/L for grad in batched_grads]
#        else:
#            images = np.asarray([datasets[node][0][index] for index in sampled_index])
#            labels = np.asarray([datasets[node][1][index] for index in sampled_index])
#            _, new_grads = Grad(model, images, labels)
#        all_grads.append(new_grads)
#
#    return all_grads

def NodeSampledIndex(nodes,datasets,batch_size):
    counter=0
    sampled_index=[]
    for node in range(len(datasets)):
        data_size = len(datasets[node][0])
        candidate_index = list(range(data_size))
        if data_size <= batch_size:
            sampled_index.append(candidate_index)
        else:
            sampled_index.append(np.random.choice(candidate_index, batch_size, replace=False))
        counter+=1
    return sampled_index



def CollectGradsAdv2(model, sampled_indices, datasets, C, delta, clipping):
    all_grads = []
    counter=0
    for node in range(len(datasets)):
        sampled_index=sampled_indices[counter]
        counter+=1
        batched_grads = []
        new_grads = []
        if clipping:
            for index in sampled_index:
                image, label = np.asarray([datasets[node][0][index]]), np.asarray([datasets[node][1][index]])
                _, grads = Grad(model, image, label)
                if delta > 0:
                    grads = ClipGradsL2(grads, C)
                elif delta == 0:
                    grads = ClipGradsL1(grads, C)
                else:
                    raise ValueError("delta is {}, it cannot be negative!".format(delta))
                    
                if not batched_grads:
                    batched_grads = grads
                else:
                    for i, grad in enumerate(grads):
                        batched_grads[i] = tf.math.add(batched_grads[i], grad)
            new_grads = [grad/L for grad in batched_grads]
        else:
            images = np.asarray([datasets[node][0][index] for index in sampled_index])
            labels = np.asarray([datasets[node][1][index] for index in sampled_index])
            _, new_grads = Grad(model, images, labels)
        all_grads.append(new_grads)

    return all_grads

def listTotensro(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg


def consensusTensors(grad,Adj_mat,cons_iter,step_cons_size):
    itCount=0
    valvec=[[[] for i in range (0,NODES)] for t in range(0,cons_iter+1)]
    if (cons_iter==0):
        valvec=grad
        return valvec
    else:
        valvec[0]=grad
    while itCount<=cons_iter-1:
        grad_int=valvec[itCount]
        if(np.mod(itCount,2)==0):
            print('Consensus iter= ',itCount+1, ' out of ',cons_iter)
        

        for node in range(0,NODES):
            nodeCountLead=node;
            diff=[];
            neighbor=np.where(Adj_mat[nodeCountLead]==1)
            neighbor=neighbor[0]
            sum_grad_neigh=[];
            tot_sum_per_dim=[[] for i in range(len(grad_int[nodeCountLead]))]
            for grad_dim in range(len(grad_int[nodeCountLead])):
                sum_grad_neigh=[];
                diff=[];
                for nodeCount in range(0,len(neighbor)): #Updating the neighbors of the leader node
#                           print(len(grad_int[nodeCountLead][grad_dim]))
                           diff=[]
#                           print(grad_int[nodeCountLead][grad_dim], ' 11111')
#                           print(grad_int[neighbor[nodeCount]][grad_dim], ' 2222')
                           diff=tf.subtract(grad_int[neighbor[nodeCount]][grad_dim], grad_int[nodeCountLead][grad_dim]) 
                           if len(sum_grad_neigh)>0:
                               sum_grad_neigh=tf.add(sum_grad_neigh,diff)
                           else:
                               sum_grad_neigh=diff
            
#                print(sum_grad_neigh, ' current Difff')
                tot_sum_per_dim[grad_dim]=tf.add(grad_int[nodeCountLead][grad_dim],tf.scalar_mul(step_cons_size, sum_grad_neigh))   
            valvec[itCount+1][nodeCountLead]= tot_sum_per_dim
         
        itCount+=1;  
    return valvec[len(valvec)-1]


def consensus(grad,Adj_mat,cons_iter,step_cons_size):
    
#    print(len(grad))
#    for subvec in grad: ## the number of subvec is equal to the number of NODES
#        print(subvec)
#    val = [[[ 0 for i in range(len(grad)) ] for k in range(len(grad[0])) ] for j in range(iterCons) ] 
#    print(grad[0].eval())
#    grad2=[grad[0][0][0]]
    
#    print(len(grad))
#    print(len(grad))
#    print(len(grad[0][0]))
#    grad_num=[[[[] for i in range(0,len(grad[0][y])) ] for y in range(0,len(grad[0]))] for k in range(0,len(grad)) ]
    grad_num=[[] for k in range(0,len(grad))]
    for i in range(0,len(grad)):
        for j in  range(0,len(grad[i])):
            for k in range(0,len(grad[i][j])):

                grad_num[i].append(grad[i][j][k].numpy().tolist())


#    orig_gradients=[[[[] for i in range(0,len(grad[0][y])) ] for y in range(0,len(grad[0]))] for k in range(0,len(grad)) ]
#    for i in range(0,len(grad)):
#        for j in  range(0,len(grad[i])):
#            for k in range(0,len(grad[i][j])):
#
#                orig_gradients[i][j][k]=(grad[i][j][k].numpy().tolist())    
    valvec=[[]]
    valvec[0][:]=grad_num
    itCount=0;
    while itCount<=cons_iter-1:
        if(np.mod(itCount,2)==0):
            print('Consensus iter= ',itCount+1, ' out of ',cons_iter)
        
        valvec.append([[] for x in range (0,len(valvec[itCount]))])
        for node in range(0,NODES):
            nodeCountLead=node;
            diff=[];
            neighbor=np.where(Adj_mat[nodeCountLead]==1)
            neighbor=neighbor[0]
            
            for nodeCount in range(0,len(neighbor)): #Updating the neighbors of the leader node
                diff2=[]
                for grads_dim_per_node in range(0,len(valvec[itCount][neighbor[nodeCount]])):
    
                    first=valvec[itCount][neighbor[nodeCount]][grads_dim_per_node]
                    second=valvec[itCount][nodeCountLead][grads_dim_per_node]
                    diff1=[]
                    if(type(first) is float):
                        first=[first]
                        second=[second]
                        
                    diff1=[x - y for x, y in zip(first, second)]
                    diff2.append(diff1)
       
                if diff:
                    counter=0
                    for elements in diff:
                        new_element= [x + y for x, y in zip(elements, diff2[counter])]
                        diff[counter]=new_element
                        counter+=1
                else:
                    diff=diff2
    
            grad_acum=[];
            for elements in diff:
    
                if grad_acum:
                    grad_acum.append(np.multiply(step_cons_size,elements).tolist())
                else:
                    grad_acum=[np.multiply(step_cons_size,elements).tolist()]
                
          
            counter=0
            new_stuff=[]
            for elements in valvec[itCount][nodeCountLead]:
    
                new_elem=np.add(elements,grad_acum[counter]).tolist()
                new_stuff.append(new_elem.copy())
                counter+=1
    
            valvec[itCount+1][nodeCountLead] =    new_stuff  
         
        itCount+=1;  
        ## Bringing the gredients back into their original shape
    cons_result= valvec[len( valvec)-1]
    out_grad=grad_num=[[[[] for i in range(0,len(grad[0][y])) ] for y in range(0,len(grad[0]))] for k in range(0,len(grad)) ]
    for i in range(0,len(grad)):
        counter=0;
        for j in  range(0,len(grad[i])):

            for k in range(0,len(grad[i][j])):
                    if(type(cons_result[i][counter]) is float):
                        cons_result[i][counter]=[cons_result[i][counter]]
                         
                    if(len(cons_result[i][counter])==1):
                        if(out_grad[i][j][k]):
                            print(np.array(out_grad[i][j][k]), ' here')
                            out_grad[i][j][k]=  (np.array(out_grad[i][j][k]).append(cons_result[i][counter][0])).tolist()
                        else:
                            out_grad[i][j][k]=  cons_result[i][counter][0]
                    else:
                        out_grad[i][j][k]= (cons_result[i][counter])

                    counter+=1;
            out_grad[i][j]=     tf.stack(out_grad[i][j])   

    return out_grad

def CombinedOriGrads2(all_grads):
    combined_grads = []

    for grads in all_grads:
        if combined_grads:
            for j in range(len(combined_grads)):
#                combined_grads[j] = combined_grads[j]+ grads[j]
                combined_grads[j] = tf.math.add(combined_grads[j], grads[j])
        else:
            combined_grads = grads.copy()
          
    for j in range(len(combined_grads)):            
        combined_grads[j]=combined_grads[j]/NODES
    return combined_grads

def CombinedOriGrads2_het(all_grads):
    combined_grads = []

    for grads in all_grads:
        if combined_grads:
            for j in range(len(combined_grads)):
#                combined_grads[j] = combined_grads[j]+ grads[j]
                combined_grads[j] = tf.math.add(combined_grads[j], grads[j])
        else:
            combined_grads = grads.copy()
          
    for j in range(len(combined_grads)):            
        combined_grads[j]=combined_grads[j]/NODES
    return combined_grads



def SameLabelSplitDataOverlap(nodes, images_by_label, labels_by_node, images_per_node=2000, overlap=0.5):
    dataset_by_node = defaultdict(list)
    current_ids = [0]*10
    candidate_pool = set(range(10))
    prev_candidates = set([])
    images_per_label = images_per_node // labels_by_node
    for node in range(nodes):
        if len(candidate_pool) <= labels_by_node:
            first_candidates = list(candidate_pool)
            remain_num_candidates = labels_by_node - len(candidate_pool)
            second_candidates = list(np.random.choice(list(prev_candidates), remain_num_candidates, replace=False))
            final_candidates = first_candidates + second_candidates
            candidate_pool = prev_candidates.difference(set(second_candidates))
            prev_candidates = set(final_candidates)
        else:
            final_candidates = list(np.random.choice(list(candidate_pool), labels_by_node, replace=False))
            candidate_pool = candidate_pool.difference(set(final_candidates))
            candidate_pool = candidate_pool.union(prev_candidates)
            prev_candidates = set(final_candidates)
        for label in final_candidates:
            current_id = current_ids[label]
            new_id = current_id + images_per_label
            if new_id > len(images_by_label[label]):
                end_id = new_id%len(images_by_label[label])
                images = images_by_label[label][current_id : new_id] + images_by_label[label][:end_id] 
                current_ids[label] = (current_id + int(overlap * images_per_label))%len(images_by_label[label])
            else:
                images = images_by_label[label][current_id : new_id]
                current_ids[label] = (current_id + int(overlap * images_per_label))
            labels = [label]*len(images)
            if len(dataset_by_node[node]) == 0:
                dataset_by_node[node].append(images)
                dataset_by_node[node].append(labels)
            else:
                dataset_by_node[node][0] += images
                dataset_by_node[node][1] += labels
    return dataset_by_node

def SameLabelSplitData(nodes, images_by_label, labels_by_node, number_of_imgs_by_node, same_num_images_per_node=False):
    dataset_by_node = defaultdict(list)
    segments = math.ceil(nodes*labels_by_node/10)
    num_of_images_by_label = [len(images_by_label[label]) for label in range(10)]
    if same_num_images_per_node:
        min_images_of_all_labels = min(num_of_images_by_label)
        num_imgs_per_segment = min(int(min_images_of_all_labels/segments), number_of_imgs_by_node // labels_by_node)
    else:
        num_imgs_per_segment = [int(num_of_images/segments) for num_of_images in num_of_images_by_label]
    segment_ids = []
    heapq.heapify(segment_ids)
    candidate_labels = list(range(10))
    np.random.shuffle(candidate_labels)
    for label in range(10):
        heapq.heappush(segment_ids, (0, label))
    for node in range(nodes):
        used_labels = []
        for _ in range(labels_by_node):
            current_id, label = heapq.heappop(segment_ids)
            actual_label = candidate_labels[label]
            if same_num_images_per_node:
                images_per_seg_per_label = num_imgs_per_segment
            else:
                images_per_seg_per_label = num_imgs_per_segment[actual_label]
            images = images_by_label[actual_label][current_id*images_per_seg_per_label
                                                   :(current_id+1)*images_per_seg_per_label]
            labels = [actual_label]*len(images)
            if len(dataset_by_node[node]) == 0:
                dataset_by_node[node].append(images)
                dataset_by_node[node].append(labels)
            else:
                dataset_by_node[node][0] += images
                dataset_by_node[node][1] += labels
            used_labels.append((current_id+1, label))
        for used_label in used_labels:
            heapq.heappush(segment_ids, used_label)
    return dataset_by_node
#
def AssignDatasets(nodes, min_labels, number_of_imgs_by_node = 2000, have_same_label_number=False, pre_process=False, same_num_images_per_node=False, sample_overlap_data=False):
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images, test_images = train_images/255.0, test_images/255.0

    if pre_process:
        train_image_samples, test_image_samples = len(train_images), len(test_images)
        original_shape = train_images[0].shape
        flatten_shape = original_shape[0]*original_shape[1]
        train_images_flatten, test_images_flatten = np.array(train_images).reshape((train_image_samples, flatten_shape)), np.array(test_images).reshape((test_image_samples, flatten_shape))
        pca_dims = PCA()
        pca_dims.fit(train_images_flatten)
        cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
        d = np.argmax(cumsum>=0.95) + 1
        pca = PCA(n_components=d)
        train_images, test_images = pca.fit_transform(train_images_flatten), pca.fit_transform(test_images_flatten)

    train_dataset = zip(train_images, train_labels)
    test_dataset = zip(test_images, test_labels)

    sorted_train_dataset = sorted(train_dataset, key=lambda x: x[1])
    sorted_test_dataset = sorted(test_dataset, key=lambda x: x[1])

    num_train_images_per_label = int(len(train_images)/nodes/min_labels)
    num_test_images_per_label = int(len(test_images)/nodes/min_labels)

    train_images_by_label = defaultdict(list)
    test_images_by_label = defaultdict(list)

    for i, image in enumerate(train_images):
        train_images_by_label[train_labels[i]].append(image)
    for i, image in enumerate(test_images):
        test_images_by_label[test_labels[i]].append(image)

    for label in range(10):
        np.random.shuffle(train_images_by_label[label])
        np.random.shuffle(test_images_by_label[label])
    
    train_dataset_by_node = defaultdict(list)
    test_dataset_by_node =defaultdict(list)

    if min_labels > 10:
        raise ValueError("Minimum number of labels is {}, which exceeds the total number of labels!".format(min_labels))
    if have_same_label_number:
        if sample_overlap_data:
            train_dataset_by_node = SameLabelSplitDataOverlap(nodes, train_images_by_label, min_labels)
            test_dataset_by_node = SameLabelSplitDataOverlap(nodes, test_images_by_label, min_labels)
        else:
            train_dataset_by_node = SameLabelSplitData(nodes, train_images_by_label, min_labels, number_of_imgs_by_node=number_of_imgs_by_node, same_num_images_per_node=same_num_images_per_node)
            test_dataset_by_node = SameLabelSplitData(nodes, test_images_by_label, min_labels, number_of_imgs_by_node=number_of_imgs_by_node, same_num_images_per_node=same_num_images_per_node)
    else:
        for label in range(min_labels):
            if label == 0:
                for node in range(nodes):
                    current_set = nodes * label + node
                    train_dataset_by_node[node].append([data[0] for data in sorted_train_dataset[current_set*num_train_images_per_label:(current_set+1)*num_train_images_per_label]])
                    train_dataset_by_node[node].append([data[1] for data in sorted_train_dataset[current_set*num_train_images_per_label:(current_set+1)*num_train_images_per_label]])
                    test_dataset_by_node[node].append([data[0] for data in sorted_test_dataset[current_set*num_test_images_per_label:(current_set+1)*num_test_images_per_label]])
                    test_dataset_by_node[node].append([data[1] for data in sorted_test_dataset[current_set*num_test_images_per_label:(current_set+1)*num_test_images_per_label]])
            else:
                for node in range(nodes):
                    current_set = nodes * label + node
                    train_dataset_by_node[node][0] += [data[0] for data in sorted_train_dataset[current_set*num_train_images_per_label:(current_set+1)*num_train_images_per_label]]
                    train_dataset_by_node[node][1] += [data[1] for data in sorted_train_dataset[current_set*num_train_images_per_label:(current_set+1)*num_train_images_per_label]]
                    test_dataset_by_node[node][0] += [data[0] for data in sorted_test_dataset[current_set*num_test_images_per_label:(current_set+1)*num_test_images_per_label]]
                    test_dataset_by_node[node][1] += [data[1] for data in sorted_test_dataset[current_set*num_test_images_per_label:(current_set+1)*num_test_images_per_label]]

    return train_dataset_by_node, test_dataset_by_node









###################### MAIN 


if __name__ == '__main__':
    time1 = time.time()
 
    Training_Accuracy = []
    Testing_Accuracy = []
    learning_rate = 0.3
    print(learning_rate)
    batch_size =20 #60000
#    train_dataset_by_node, test_dataset_by_node = AssignDatasets2(NODES)
    num_labels_per_node=1
    train_dataset_by_node, test_dataset_by_node = AssignDatasets(NODES, min_labels = num_labels_per_node, number_of_imgs_by_node = 60000/NODES, have_same_label_number=False, pre_process=False, same_num_images_per_node=True, sample_overlap_data=False)
    Training_Accuracy.append([])
    Testing_Accuracy.append([])


    
    
    model_cons_1=CreateModel(np.shape(train_dataset_by_node[0][0][0]))
    model_cons_2=CreateModel(np.shape(train_dataset_by_node[0][0][0]))
    model_cons_3=CreateModel(np.shape(train_dataset_by_node[0][0][0]))
    model_cons_4=CreateModel(np.shape(train_dataset_by_node[0][0][0]))
    model_0 =CreateModel(np.shape(train_dataset_by_node[0][0][0]))
    model_cons_1.set_weights(model_0.get_weights())
    model_cons_2.set_weights(model_0.get_weights())
    model_cons_3.set_weights(model_0.get_weights())
    model_cons_4.set_weights(model_0.get_weights())
 
    train_accuracy_results = []
    train_loss_results=[]
    test_loss_results=[]
    test_accuracy_results = []
    
    train_accuracy_results_cons_1 = []
    train_loss_results_cons_1=[]
    test_loss_results_cons_1=[]
    test_accuracy_results_cons_1 = []
    
    train_accuracy_results_cons_2 = []
    train_loss_results_cons_2=[]
    test_loss_results_cons_2=[]
    test_accuracy_results_cons_2 = []
    
    train_accuracy_results_cons_3 = []
    train_loss_results_cons_3=[]
    test_loss_results_cons_3=[]
    test_accuracy_results_cons_3 = []
    
    train_accuracy_results_cons_4 = []
    train_loss_results_cons_4=[]
    test_loss_results_cons_4=[]
    test_accuracy_results_cons_4 = []
    num_epoches = 150


#                        
    for epoch in range(num_epoches):
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_accuracy_1 = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_accuracy_2 = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_accuracy_3 = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_accuracy_4 = tf.keras.metrics.SparseCategoricalAccuracy()
        if(epoch!=0 and epoch %10 ==0 and learning_rate>=0.001):
            learning_rate = learning_rate/1.1
        
        optimizer = SetOptimizer(learning_rate)
        optimizer1 = SetOptimizer(learning_rate)
        optimizer2= SetOptimizer(learning_rate)
        optimizer3 = SetOptimizer(learning_rate)
        optimizer4 = SetOptimizer(learning_rate)
        data_size = len(train_dataset_by_node[1][0])
        
#        candidate_index = list(range(data_size))
#        L = batch_size
#        if data_size <= batch_size:
#            sampled_index = candidate_index
#            L = data_size
#        else:
#            sampled_index = np.random.choice(candidate_index, batch_size, replace=False)
        sampled_index=NodeSampledIndex(NODES,train_dataset_by_node,batch_size)    
        all_grads_0 = CollectGradsAdv2(model_0, sampled_index, train_dataset_by_node, C=1000, delta=0, clipping=False)  ## Does the SGD with batch size and colelcts all the gradients
        all_grads_1 = CollectGradsAdv2(model_cons_1, sampled_index, train_dataset_by_node, C=1000, delta=0, clipping=False)
        all_grads_2 = CollectGradsAdv2(model_cons_2, sampled_index, train_dataset_by_node, C=1000, delta=0, clipping=False)
        all_grads_3 = CollectGradsAdv2(model_cons_3, sampled_index, train_dataset_by_node, C=1000, delta=0, clipping=False)
        all_grads_4 = CollectGradsAdv2(model_cons_4, sampled_index, train_dataset_by_node, C=1000, delta=0, clipping=False)
#        print(all_grads_0[1], ' GRAD 0')
#        print(all_grads_1[1], ' GRAD 1')
#        hetero = False


        sto_grads=CombinedOriGrads2(all_grads_0)
        optimizer.apply_gradients(zip(sto_grads, model_0.trainable_variables))

        
               
        accuracy = 0
        loss_train=0
        for node in range(NODES):
          accuracy += float(epoch_accuracy(np.asarray(train_dataset_by_node[node][1]), model_0(np.asarray(train_dataset_by_node[node][0]))))
          loss_train+=float(Loss(model_0,np.asarray(train_dataset_by_node[node][0]),np.asarray(train_dataset_by_node[node][1])))

        accuracy /= NODES
        loss_train /=NODES
        train_accuracy_results.append(accuracy)
        train_loss_results.append(loss_train)
    
        
        test_accuracy = 0
        loss_test=0
        for node in range(NODES):
            test_accuracy += float(epoch_accuracy(np.asarray(test_dataset_by_node[node][1]), model_0(np.asarray(test_dataset_by_node[node][0]))))
            loss_test += float(Loss(model_0,np.asarray(test_dataset_by_node[node][0]), np.asarray(test_dataset_by_node[node][1])))
        test_accuracy /= NODES
        loss_test /= NODES
        test_accuracy_results.append(test_accuracy)
        test_loss_results.append(loss_test)
        
        iterCons_1=5
        iterCons_2=10
        iterCons_3=20
        
        iterCons_4=20
        if epoch!=0:
            if (np.mod(epoch,8)==0):
                iterCons_11=int((iterCons_11)+2)
            if (np.mod(epoch,4)==0):   
                iterCons_22=int((iterCons_22)+2)
            if (np.mod(epoch,2)==0):   
                iterCons_33=int((iterCons_33)+2)
        else:
                iterCons_11=iterCons_1
                iterCons_22=iterCons_2
                iterCons_33=iterCons_3
        
        iterCons_11=min(iterCons_11,3)
        iterCons_22=min(iterCons_22,8)
        iterCons_33=min(iterCons_33,20)
        step_cons=0.06; #0.005
#        

##################### OLD Style (too SLOW !!) Doing all the stuff in the algebraic manner with expanding the tensors and treatign them as matrices
###        cons_value_vec_1=consensus(all_grads_1,Adj_mat,iterCons_1,step_cons)
###       cons_value_vec_2=consensus(all_grads_2,Adj_mat,iterCons_2,step_cons)
###       cons_value_vec_33=consensus(all_grads_3,Adj_mat,iterCons_3,step_cons)
###       cons_value_vec_44=consensus(all_grads_4,Adj_mat,iterCons_4,step_cons)
#####################

##################### USING TF math operators (much faster)
        cons_value_vec_1=consensusTensors(all_grads_1,Adj_mat,iterCons_11,step_cons)
        cons_value_vec_2=consensusTensors(all_grads_2,Adj_mat,iterCons_22,step_cons)
        cons_value_vec_3=consensusTensors(all_grads_3,Adj_mat,iterCons_33,step_cons)
        cons_value_vec_4=consensusTensors(all_grads_4,Adj_mat,iterCons_4,step_cons)
        
####################        
        sampled_node_index=np.random.randint(0,NODES-1)
        cons_value_vec2_1=cons_value_vec_1[sampled_node_index] 
        cons_value_vec2_2=cons_value_vec_2[sampled_node_index]  
        cons_value_vec2_3=cons_value_vec_3[sampled_node_index]  
        cons_value_vec2_4=cons_value_vec_4[sampled_node_index]  
#
        for i in range(len(cons_value_vec2_1)):

            cons_value_vec2_1[i]=tf.dtypes.cast(cons_value_vec2_1[i], tf.float64)
            cons_value_vec2_2[i]=tf.dtypes.cast(cons_value_vec2_2[i], tf.float64)
            cons_value_vec2_3[i]=tf.dtypes.cast(cons_value_vec2_3[i], tf.float64)
            cons_value_vec2_4[i]=tf.dtypes.cast(cons_value_vec2_4[i], tf.float64)

        optimizer.apply_gradients(zip(cons_value_vec2_1, model_cons_1.trainable_variables))
        optimizer2.apply_gradients(zip(cons_value_vec2_2, model_cons_2.trainable_variables))
        optimizer3.apply_gradients(zip(cons_value_vec2_3, model_cons_3.trainable_variables))
        optimizer4.apply_gradients(zip(cons_value_vec2_4, model_cons_4.trainable_variables))
        accuracy_1 = 0
        loss_train_1=0
        accuracy_2 = 0
        loss_train_2=0
        accuracy_3 = 0
        loss_train_3=0
        accuracy_4 = 0
        loss_train_4=0
        for node in range(NODES):
          accuracy_1 += float(epoch_accuracy_1(np.asarray(train_dataset_by_node[node][1]), model_cons_1(np.asarray(train_dataset_by_node[node][0]))))
          loss_train_1+=float(Loss(model_cons_1,np.asarray(train_dataset_by_node[node][0]),np.asarray(train_dataset_by_node[node][1])))
          accuracy_2 += float(epoch_accuracy_2(np.asarray(train_dataset_by_node[node][1]), model_cons_2(np.asarray(train_dataset_by_node[node][0]))))
          loss_train_2+=float(Loss(model_cons_2,np.asarray(train_dataset_by_node[node][0]),np.asarray(train_dataset_by_node[node][1])))
          accuracy_3 += float(epoch_accuracy_3(np.asarray(train_dataset_by_node[node][1]), model_cons_3(np.asarray(train_dataset_by_node[node][0]))))
          loss_train_3+=float(Loss(model_cons_3,np.asarray(train_dataset_by_node[node][0]),np.asarray(train_dataset_by_node[node][1])))
          accuracy_4 += float(epoch_accuracy_4(np.asarray(train_dataset_by_node[node][1]), model_cons_4(np.asarray(train_dataset_by_node[node][0]))))
          loss_train_4+=float(Loss(model_cons_4,np.asarray(train_dataset_by_node[node][0]),np.asarray(train_dataset_by_node[node][1])))
        
        accuracy_1 /= NODES
        accuracy_2 /= NODES
        accuracy_3 /= NODES
        accuracy_4 /= NODES
        loss_train_1 /=NODES
        loss_train_2 /=NODES
        loss_train_3 /=NODES
        loss_train_4 /=NODES
        train_accuracy_results_cons_1.append(accuracy_1)
        train_loss_results_cons_1.append(loss_train_1)  
        train_accuracy_results_cons_2.append(accuracy_2)
        train_loss_results_cons_2.append(loss_train_2)  
        train_accuracy_results_cons_3.append(accuracy_3)
        train_loss_results_cons_3.append(loss_train_3)  
        train_accuracy_results_cons_4.append(accuracy_4)
        train_loss_results_cons_4.append(loss_train_4)  
        
        test_accuracy_1 = 0
        loss_test_1=0
        test_accuracy_2 = 0
        loss_test_2=0
        test_accuracy_3 = 0
        loss_test_3=0
        test_accuracy_4 = 0
        loss_test_4=0
        for node in range(NODES):
            test_accuracy_1 += float(epoch_accuracy_1(np.asarray(test_dataset_by_node[node][1]), model_cons_1(np.asarray(test_dataset_by_node[node][0]))))
            loss_test_1 += float(Loss(model_cons_1,np.asarray(test_dataset_by_node[node][0]), np.asarray(test_dataset_by_node[node][1])))
            test_accuracy_2 += float(epoch_accuracy_2(np.asarray(test_dataset_by_node[node][1]), model_cons_2(np.asarray(test_dataset_by_node[node][0]))))
            loss_test_2 += float(Loss(model_cons_2,np.asarray(test_dataset_by_node[node][0]), np.asarray(test_dataset_by_node[node][1])))
            test_accuracy_3 += float(epoch_accuracy_3(np.asarray(test_dataset_by_node[node][1]), model_cons_3(np.asarray(test_dataset_by_node[node][0]))))
            loss_test_3 += float(Loss(model_cons_3,np.asarray(test_dataset_by_node[node][0]), np.asarray(test_dataset_by_node[node][1])))
            test_accuracy_4 += float(epoch_accuracy_4(np.asarray(test_dataset_by_node[node][1]), model_cons_4(np.asarray(test_dataset_by_node[node][0]))))
            loss_test_4 += float(Loss(model_cons_4,np.asarray(test_dataset_by_node[node][0]), np.asarray(test_dataset_by_node[node][1])))
        
        test_accuracy_1 /= NODES
        loss_test_1 /= NODES
        test_accuracy_2 /= NODES
        loss_test_2 /= NODES
        test_accuracy_3 /= NODES
        loss_test_3 /= NODES
        test_accuracy_4 /= NODES
        loss_test_4 /= NODES
        test_accuracy_results_cons_1.append(test_accuracy_1)
        test_loss_results_cons_1.append(loss_test_1)
        test_accuracy_results_cons_2.append(test_accuracy_2)
        test_loss_results_cons_2.append(loss_test_2)
        test_accuracy_results_cons_3.append(test_accuracy_3)
        test_loss_results_cons_3.append(loss_test_3)
        test_accuracy_results_cons_4.append(test_accuracy_4)
        test_loss_results_cons_4.append(loss_test_4)
        
        if epoch % 2 == 0:

            print("Iter {:03d}: Training Accuracy: {:.3%}".format(epoch, accuracy))
#            print("Iter {:03d}: Testing Accuracy: {:.3%}".format(epoch, test_accuracy))
#            print("Iter {:03d}: Training loss: {:}".format(epoch, loss_train))
#            print("Iter {:03d}: Testing loss: {:}".format(epoch, loss_test))            
#            print("Iter {:03d}: learning_rate: {:}".format(epoch, learning_rate))
            

            print("Iter {:03d}: Training Accuracy Cons1: {:.3%}".format(epoch, accuracy_1))
            print("Iter {:03d}: Training Accuracy Cons2: {:.3%}".format(epoch, accuracy_2))
            print("Iter {:03d}: Training Accuracy Cons3: {:.3%}".format(epoch, accuracy_3))
            print("Iter {:03d}: Training Accuracy Cons4: {:.3%}".format(epoch, accuracy_4))
            
#            print("Iter {:03d}: Testing Accuracy Cons1: {:.3%}".format(epoch, test_accuracy_1))
#            print("Iter {:03d}: Testing Accuracy Cons2: {:.3%}".format(epoch, test_accuracy_2))
#            print("Iter {:03d}: Testing Accuracy Cons3: {:.3%}".format(epoch, test_accuracy_3))
#            print("Iter {:03d}: Testing Accuracy Cons4: {:.3%}".format(epoch, test_accuracy_4))
#            
#            print("Iter {:03d}: Training loss Cons: {:}".format(epoch, loss_train_2))
#            print("Iter {:03d}: Testing loss Cons: {:}".format(epoch, loss_test_3))            
#            print("Iter {:03d}: learning_rate Cons: {:}".format(epoch, learning_rate))
            
            
    np.save('Tappared_'+ '_Het_OutputFile_FL_'     + str(num_epoches) +'_nodes_'+ str(NODES)+'_number_labels_per_node_'+str(num_labels_per_node),[num_epoches,train_accuracy_results,train_loss_results,test_accuracy_results,test_loss_results,batch_size])
    np.save('Tappared_'+  '_Het_OutputFile_Cons_1_' + str(iterCons_1) + '_nodes_'+ str(NODES)+'_number_labels_per_node_'+str(num_labels_per_node),[num_epoches,train_accuracy_results_cons_1,train_loss_results_cons_1,test_accuracy_results_cons_1,test_loss_results_cons_1,batch_size])
    np.save('Tappared_'+ '_Het_OutputFile_Cons_2_' + str(iterCons_2) + '_nodes_'+ str(NODES)+'_number_labels_per_node_'+str(num_labels_per_node),[num_epoches,train_accuracy_results_cons_2,train_loss_results_cons_2,test_accuracy_results_cons_2,test_loss_results_cons_2,batch_size])
    np.save('Tappared_'+ '_Het_OutputFile_Cons_3_' + str(iterCons_3) + '_nodes_'+ str(NODES)+'_number_labels_per_node_'+str(num_labels_per_node),[num_epoches,train_accuracy_results_cons_3,train_loss_results_cons_3,test_accuracy_results_cons_3,test_loss_results_cons_3,batch_size])
    np.save('Tappared_'+ '_Het_OutputFile_Cons_4_' + str(iterCons_4) + '_nodes_'+ str(NODES)+'_number_labels_per_node_'+str(num_labels_per_node),[num_epoches,train_accuracy_results_cons_4,train_loss_results_cons_4,test_accuracy_results_cons_4,test_loss_results_cons_4,batch_size])
    time2 = time.time()
    print(time2-time1)
    print(Training_Accuracy)
    print(Testing_Accuracy)
