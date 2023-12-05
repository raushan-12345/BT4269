#!/usr/bin/env python
# coding: utf-8

# In[30]:


pip install torchvision


# # import the libraries

# In[31]:


import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


# # Load pretrained model

# In[32]:


# Load the pretrained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Class labels from official PyTorch documentation for the pretrained model
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'Bus','Tractor',
    'truck','Auto','Tempo Traveller','van','boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


# # Getting the predictions

# In[33]:


def get_prediction(img_path, threshold):
    # Open the image
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)

    # Get predictions from the model
    with torch.no_grad():
        pred = model([img])

    # Filter predictions based on threshold
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return pred_boxes, pred_class


# # Performing object detection and displaying the results

# In[34]:



def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=1, text_th=3):
    # Get predictions
    boxes, pred_cls = get_prediction(img_path, threshold)

    # Open the image with OpenCV
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Loop over the predictions and draw bounding boxes
    for i in range(len(boxes)):
        box = boxes[i]
        pred_class = pred_cls[i]

        # Extract coordinates for drawing the rectangle and text
        x1, y1, x2, y2 = box

        # Convert the coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw the rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=rect_th)

        # Draw the class label text above the rectangle
        cv2.putText(img, pred_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# # Calling the object detection RCNN model

# In[35]:


img_path = "C:\\Users\\raush\\Downloads\\object detection\\object detection\\Dataset\\motor cycle\\1570811001026.jpeg"
object_detection_api(img_path, threshold=0.8)


# In[36]:


img_path = "C:\\Users\\raush\\Downloads\\object detection\\object detection\\Dataset\\Auto\\Datacluster Auto (111).jpg"
object_detection_api(img_path, threshold=0.8)


# In[37]:


img_path = "C:\\Users\\raush\\Downloads\\object detection\\object detection\\Dataset\\car\\vid_4_9880.jpg"
object_detection_api(img_path, threshold=0.8)


# In[38]:


img_path = "C:\\Users\\raush\\Downloads\\object detection\\object detection\\Dataset\\Bus\\20210522_06_38_14_000_RMkLr8rWJDMBfnDv6WRtqfoMLT83_F_3264_2448.jpg"
object_detection_api(img_path, threshold=0.5,text_size=3)


# # Conclusion:
# This code implements an object detection API using the Faster R-CNN model. It takes an input image, detects objects in the image, and displays the image with bounding boxes and class labels for the detected objects. 

# In[ ]:





# In[ ]:




