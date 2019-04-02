# coding=utf-8
import imageio
import argparse
import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.5):
    """Filter YOLO boxes by thresholding on object and class confidence.
    Argument:
    box_confidence--tensor of shape(19,19,5,1)
    boxes--tensor of shape(19,19,5,4)
    box_class_probs--tensor of shape(19,19,5,1)
    threshold--real value,if[highest class probability score<threshol],then get rid of the corresponding box.
    """
    # step 1:Compute box scores
    box_scores = box_confidence * box_class_probs#shape=(19,19,5,80)
    
    # Step 2:Find the box_classes(index) and box_class_scores(scores)
    box_classes = K.argmax(box_scores, axis=-1)  # return index,shape=(361,5,1),-1 represents the last dimension
    box_class_scores = K.max(box_scores, axis=-1, keepdims=False)  # return maximum of scores in every box
    
    # Step 3:Create a filtering mask based on 'box_class_scores' by using 'threshold'.
    filter_mask = box_class_scores >= threshold  # every element will be compared with threshold,shape=(19,19,5,1)
    
    # Step 4:Apply the mask to score,boxes,and classes
    scores = tf.boolean_mask(box_class_scores, filter_mask)
    boxes = tf.boolean_mask(boxes, filter_mask)
    classes = tf.boolean_mask(box_classes, filter_mask)
    
    return scores, boxes, classes
# with tf.Session() as sess:
#     box_confidence=tf.random_normal([19,19,5,1],mean=1,stddev=4,seed=1)
#     boxes=tf.random_normal([19,19,5,4],mean=1,stddev=4,seed=1)
#     box_class_probs=tf.random_normal([19,19,5,80],mean=1,stddev=4,seed=1)
#     scores,boxes,classes=yolo_filter_boxes(box_confidence,boxes,box_class_probs,threshold=.5)
#     
#     print('scores[2]=',scores[20].eval())
#     print('boxes[2]=',boxes[20].eval())
#     print('classes[2]=',classes[20].eval())
#     print('scores.shape=',scores.shape)
#     print('boxes.shape=',boxes.shape)
#     print('classes.shape=',classes.shape)

    
def iou(box1, box2):
    """Implement the intersection over union(Iou)between box1 and box2"""
    # Calculate the coordinates of the intersection of box1 and box2,Area
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    Area_intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate the union
    Area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    Area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    Area_union = Area1 + Area2 - Area_intersection
    
    # Calculate Iou 
    iou = Area_intersection / Area_union 
    
    return iou
# #Test
# box1 = (2, 1, 4, 3)
# box2 = (1, 2, 3, 4) 
# print("iou = " + str(iou(box1, box2)))      


# Yolo_non_max_supression
def yolo_non_max_supression(scores, boxes, classes, max_boxes=20, iou_threshold=0.5):
    """Applies Non_max suppression(NMS) to set of boxes
    Arguments:
    scores--tensor of shape(None,)output of yolo_filter_boxes()
    boxes--tensor of shape(None,4)
    max_boxes--integer,maximum number of predicted boxes you'd like
    """
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable ma_boxes_tensor 
    
    # get the list of indices corresponding to the boxes to keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold, name=None)
    
    # Use K.gather() to select only nms_indices from scores,boxes,and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes
# #Test program
# with tf.Session() as sess:
#     scores=tf.random_normal([54,],mean=1,stddev=4,seed=1)
#     boxes=tf.random_normal([54,4],mean=1,stddev=4,seed=1)
#     classes=tf.random_normal([54,],mean=1,stddev=4,seed=1)
#     scores,boxes,classes=yolo_non_max_supression(scores, boxes, classes)
#     print('scores[2]='+str(scores[2].eval()))
#     print('boxes[2]=',boxes[2].eval())
#     print('classes[2]=',classes[2].eval())
#     print("scores.shape = " + str(scores.eval().shape))
#     print("boxes.shape = " + str(boxes.eval().shape))
#     print("classes.shape = " + str(classes.eval().shape))


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=50, score_threshold=.2, iou_threshold=.7):
    """
    Argument:
    yolo_outputs--output of your the encoding model contains 4 tensor:
    box_confidence: tensor of shape(None,19,19,5,1)
    box_xy:tensor of shape(None,19,19,5,2)
    box_wh:tensor of shape(None,19,19,5,2)
    box_class_probs:tensor of shape(None,19,19,5,80)
    """
    # Retrieve outputs of YOLO model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    
    # Convert the way of representing boxes to which seem as (x1,x2,y1,y2)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    # Perform Score-filtering with a threshold of score
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)
    
    # Scale boxes back to original image shape
    boxes = scale_boxes(boxes, image_shape)
    
    # Perform Non_max_supression with a threshold of Iou
    scores, boxes, classes = yolo_non_max_supression(scores, boxes, classes, max_boxes, iou_threshold)
    
    return scores, boxes, classes
# #Test program
# with tf.Session() as test_b:
#     yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
#                     tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
#                     tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
#                     tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
#     scores, boxes, classes = yolo_eval(yolo_outputs)
#     print("scores[2] = " + str(scores[2].eval()))
#     print("boxes[2] = " + str(boxes[2].eval()))
#     print("classes[2] = " + str(classes[2].eval()))
#     print("scores.shape = " + str(scores.eval().shape))
#     print("boxes.shape = " + str(boxes.eval().shape))
#     print("classes.shape = " + str(classes.eval().shape))


# Test YOLO pretrained model on images
# Create a session to start your graph
sess = K.get_session()

# loading class_name,anchors
class_names = read_classes('model_data/coco_classes.txt')
anchors = read_anchors('model_data/yolo_anchors.txt')
image_shape = (720., 1280.)

# Loading a pre_trained model which's shape of output is (m,19,19,425)
yolo_model = load_model('model_data/yolo.h5')
#Show detail of the model
yolo_model.summary()

# Convert output of the model to usable bounding box tensors (m,19,19,5,85)-->(m,19,19,5,1)、(m,19,19,5,4)、(m,19,19,5,80)
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

# Filter boxes
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


# Define a function of prediction which runs the graph to test YOLO on an image
def predict(sess, image_file):
    
    # Pre_process image
    image, image_data = preprocess_image('images/' + image_file, model_image_size=(608, 608))
    
    # Run the sess with the correct tensors and choose the correct placeholder in the feed_dict
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input:image_data, K.learning_phase():0})
    
    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes
    colors = generate_colors(class_names)
    # Draw bounding boxes on image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join('output_image', image_file), quality=90)
    # Display the results in IDE
    output_image = imageio.imread(os.path.join('output_image', image_file))
    imshow(output_image)
    plt.show()
    
    return out_scores, out_boxes, out_classes

    
# Implement the program
out_scores, out_boxes, out_classes = predict(sess, 'test1.png')
      
'''
Created on 2019年3月23日

@author: Administrator
'''
