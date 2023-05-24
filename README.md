# HAR
I developed a human activity recognisation system using yolov6 model trainrd on custom dataset of human activity with labeling.


# ABOUT THE SYSTEM
 
ABOUT THE SYSTEM

	Human Activity Recognition (HAR) system is a type of machine learning technology that is designed of automatically identify and classify different human activities. 
	The system typically involves capturing video footage of a person performing various activities, such as walking, running, or sitting, and using machine learning algorithms to analyze the video and identify specific movements.
	The system may use various techniques for image processing and feature extraction, such as background subtraction, edge detection, and motion analysis.
	Once the features have been extracted, the system can use machine learning algorithms to classify the activity being performed.
	HAR systems can be used in a variety of Sectors, such as – 
o	In an Offices
o	In a Homes
o	In a Shops/Malls
o	In a Public Places like – Stations, Airports, etc.
o	At Borders to help Army
Activities: - 
	Only Person:		
o	Cycling
o	Pushing
o	Pulling
o	Using Phone
o	Using Laptop
o	Drinking
o	Eating
o	Reading

	Person with Object:		
o	Standing
o	Sitting
o	Sleeping
o	Laughing
o	Walking
o	Running
o	Stair up
o	Stair down
o	Clapping
o	Jumping

	Two or More Person:	
o	Hand-Shaking
o	Hugging





# FEATURES OF THE USED TOOLS





FEATURE OF TOOLS USED

	Google Colab: -
Google Colab is a cloud-based platform for running Python code in a Jupyter notebook environment. It is provided by Google for free, and allows users to write and execute Python code in a web browser without the need for any additional setup or installation. Colab offers access to powerful hardware resources like GPUs and TPUs for machine learning tasks, pre-installed packages and libraries, and collaboration and sharing capabilities with others.

	Python: -
Python is a high-level, interpreted programming language that is known for its simple syntax and ease of use. It was first released in 1991 and has since become one of the most widely used programming languages in the world. Python can be used for a wide range of applications, including web development, scientific computing, data analysis, and machine learning. It is an open-source language with a large and active community of developers, who have created numerous libraries and frameworks that extend its capabilities.

	Common Uses of Python: -

•	Web Development: Python can be used to create web applications using frameworks like Django and Flask.

•	Data Science and Analytics: Python is a popular language for data analysis and visualization, with libraries like Pandas and Matplotlib.

•	Machine Learning and Artificial Intelligence: Python is used extensively in machine learning and AI, with libraries like PyTorch, TensorFlow and Keras.

•	Scripting: Python is commonly used as a scripting language for automation tasks, such as system administration and testing.

•	Game Development: Python can be used to create games using libraries like Pygame and PyOpenGL.

FEATURE OF TOOLS USED


•	Education: Python is an excellent language for teaching programming concepts, and is widely used in introductory programming courses.

•	Desktop Applications: Python can be used to create desktop applications using frameworks like PyQt and wxPython.


	Characteristics Python: -
  
•	Simple and easy to learn: Python has a simple and easy-to-understand syntax, making it an ideal language for beginners to learn programming.

•	Versatile: Python is a versatile language that can be used for a wide range of applications, from web development to machine learning and artificial intelligence.

•	Interpreted: Python is an interpreted language, which means that it does not require compilation before execution. This makes it faster to write and test code.

•	High-level: Python is a high-level language, which means that it abstracts away many of the lower-level details of the computer's hardware, making it easier to write code.

•	Object-oriented: Python is an object-oriented language, which means that it allows for the creation and manipulation of objects, making it easier to organize and structure code.

•	Open source: Python is an open-source language, which means that its source code is freely available for anyone to use, modify, and distribute.

•	Large standard library: Python comes with a large standard library, which includes modules for common tasks such as web development, data analysis, and cryptography.



FEATURE OF TOOLS USED

	Python Libraries:


1)	PyTorch: YOLOv6 is built on top of PyTorch, a popular deep learning framework. PyTorch provides the necessary tools and modules to train and deploy YOLOv6 on custom datasets.

2)	NumPy: NumPy is a powerful numerical computing library that is used for handling large arrays and matrices of data. It is used in YOLOv6 for data preprocessing and postprocessing.

3)	OpenCV: OpenCV is an open-source computer vision library that provides tools for image and video processing. It is used in YOLOv6 for data augmentation, visualization, and processing.

4)	Tqdm: Tqdm is a progress bar library that can be used to display progress bars during long-running tasks, such as training YOLOv6 on large datasets.
5)	Tensorboard: Tensorboard is a visualization library that is used for monitoring and visualizing the training progress of deep learning models.

6)	Flask: Flask is use to build a RESTful API that exposes the YOLOv6 model as a service, allowing other applications to access the model and perform object detection.


	Flask: - 

Flask is a web framework for building web applications using the Python programming language. It provides with a simple and lightweight way to create web applications that can handle user requests and responses.

Flask allows to define routes, which are URLs that map to specific functions that handle user requests. These functions can then generate dynamic content, such as HTML pages, based on the user's input.

Flask also supports libraries that provide additional functionality to the framework, such as support for database access, authentication, and session management. This makes it easy for developers to add complex features to their web applications without having to write everything from scratch.



FEATURE OF TOOLS USED


	What is YOLO: 

YOLO stands for You Only Look Once, which refers to the fact that the object detection algorithm in YOLO models processes the entire image only once to detect objects, unlike other traditional object detection systems that use multiple regions of interest (ROIs) to detect objects.

In other words, YOLO models perform object detection in a single forward pass of the neural network, making them more efficient and faster than other object detection systems. This is achieved by dividing the image into a grid and applying a single neural network to predict bounding boxes and class probabilities for objects in each grid cell. The predictions are then post-processed to eliminate duplicate detections and refine the bounding box locations.

This approach allows YOLO models to achieve high accuracy and real-time performance, making them popular in a variety of applications such as autonomous vehicles, surveillance systems, and robotics.



	What is a YOLOv6:

The main aim of YOLOv6 is to propagate its use for industrial applications.

YOLOv6 (You Only Look Once) version 6 is an advanced deep learning-based object detection system. It is a state-of-the-art system for real-time object detection, which can detect objects in an image or video stream with high accuracy and fast speed.

 YOLOv6 is based on a single convolutional neural network, which is trained to recognize and locate objects in an image or video frame. One of the key features of YOLOv6 is its ability to detect multiple objects simultaneously, making it an ideal choice for many object detection tasks.


	YOLOv6 Architecture:

The YOLOv6 architecture consists of three main parts: Backbone, Neck, and Head.



FEATURE OF TOOLS USED




•	Backbone:
The backbone is responsible for extracting high-level features from input images. The backbone network in YOLOv6 is based on the EfficientRep Backbone architecture, which is a highly efficient and scalable convolutional neural network. The EfficientRep Backbone architecture includes several key features that make it well-suited for object detection tasks:

–	Convolutional layers with larger receptive fields to capture more context in the input images.

–	Residual connections between convolutional blocks to help propagate gradients and improve training efficiency.

–	Cross-stage partial connections (CSP) that connect early and late layers in the network, enabling better information flow and feature reuse.

–	Squeeze-and-excitation (SE) blocks, which dynamically recalibrate channel-wise feature responses to improve the discriminative power of the network.

•	Neck:
The neck of YOLOv6 is a set of intermediate layers that connect the backbone to the head of the network. The neck includes several important modules, including:


–	Spatial Pyramid Pooling (SPP): This module enables the network to operate on feature maps of different sizes, making it more effective in detecting objects at different scales. The SPP module applies pooling operations with different kernel sizes to the same feature map, generating a set of fixed-length representations that capture information at different scales.


FEATURE OF TOOLS USED

–	Path Aggregation Network (PANet): This module fuses features from different layers in the backbone to generate high-quality object detection results. The PANet module includes a top-down pathway and a bottom-up pathway, which work together to aggregate information from different scales and locations in the feature maps.

–	Convolutional layers: The neck also includes several convolutional layers that further process the features from the backbone and prepare them for the head of the network.


•	Head:
The head of YOLOv6 is responsible for predicting the bounding boxes and class probabilities of objects in the input image. The head includes several key components, including:

–	Convolutional layers: The head includes a series of convolutional layers that transform the features from the neck into a set of features that can be used to predict object detections.

–	Global average pooling: The output of the convolutional layers is passed through a global average pooling layer, which generates a fixed-length representation of the features.

–	Fully connected layers: The pooled features are then passed through a set of fully connected layers, which predict the class probabilities and bounding boxes of objects in the input image.

–	Loss function: YOLOv6 uses a focal loss function, which focuses on hard examples during training to improve the network's ability to detect objects under challenging conditions.

•	Anchor-free detection method:
Anchor-free object detection is a type of object detection method that eliminates the use of anchor boxes.

–	In anchor-based object detection models, anchor boxes are predefined boxes of fixed sizes and aspect ratios that are used as references to predict the location and size of objects in the image. The model predicts the offsets from these anchor boxes to localize the objects in the image. However, this can be challenging when dealing with objects of varying sizes and shapes, and requires careful tuning of the anchor box sizes and aspect ratios.


FEATURE OF TOOLS USED

–	To overcome limitations of anchor-based detection methods, anchor-free object detection methods directly predict the bounding box parameters from the features of the image without using anchor boxes. These methods use a heatmap-based approach to locate objects in the image. The model predicts a heatmap for each object class, where each pixel in the heatmap represents the likelihood of an object center being present in that location. The model then predicts the offsets from the object center to the object boundary to define the object's bounding box.


•	Loss functions:
In YOLOv6, VFL (Varifocal Loss) is used as the classification loss function, while DFL (Distribution Focal Loss) is used in combination with SIoU (Scale-Invariant IoU) or GIoU (Generalized IoU) as the box regression loss function.

–	VFL is a modification of the traditional cross-entropy loss function that considers the imbalance in the number of samples between different classes. It weights each sample differently based on its classification difficulty, with harder-to-classify samples receiving higher weights. This approach helps to reduce the influence of easy-to-classify samples on the training process and improve the model's overall accuracy.

–	DFL is another modification of the cross-entropy loss function that addresses the issue of class imbalance by introducing a distribution parameter that controls the weighting of each class. The distribution parameter is learned during training and helps to balance the contribution of each class to the loss function.

–	SIoU and GIoU are regression loss functions that calculate the distance between the predicted bounding box and the ground-truth bounding box. They are scale-invariant, which means that they can handle objects of different sizes and aspect ratios. By using SIoU or GIoU in combination with DFL, the model can learn to predict accurate bounding box coordinates while taking into account the distribution of objects in the image.


	Advantages: -

•	Improved accuracy: YOLOv6 achieves state-of-the-art performance on several object detection benchmarks, outperforming Faster R-CNN and other YOLO models in terms of both accuracy and speed.


FEATURE OF TOOLS USED

•	Faster inference: YOLOv6 is designed to be highly efficient and scalable, allowing for fast inference on a wide range of hardware platforms. YOLOv6 can process images in real-time on a CPU, and can achieve even faster inference on a GPU.
•	Customizability: YOLOv6 is highly customizable and can be adapted to a wide range of object detection tasks. It supports a wide range of input resolutions, enabling it to detect objects at different scales and resolutions.

•	Easy to use: YOLOv6 is relatively easy to use compared to other object detection frameworks. It is built on the PyTorch framework, which makes it easy to train and deploy models on a wide range of hardware platforms.

•	Improved features: YOLOv6 introduces several new features that improve its accuracy and efficiency, such as the use of cross-stage partial connections, spatial pyramid pooling, and path aggregation network modules.



 



# SYSTEM DOCUMENTATION





SYSTEM DOCUMENTATION
	Time Line Chart: -
 

SYSTEM DOCUMENTATION
	System Flow Chart: -
 
SYSTEM DOCUMENTATION

	Dataset: - 
We named “HAR image dataset” to our dataset.
•	Source:
Our dataset is combined of four datasets,
–	We have manually gathered images from internet and manually labelled them using software called ‘Labelimg.’ This software gives exact label file as we want.

–	We extract part of okutama-action dataset as per our requirement, got from website http://okutama-action.org/ .  label format of this dataset is different so we create python script to convert them automatically into our label format.

–	We found another dataset from website https://vision.cs.uiuc.edu/projects/activity/ . Labelling of this dataset is also different from both our required format and okutama-action dataset format. We do changes in script and convert all labels.

–	Another dataset we found from https://kaggle.com/ called “Cyclist Dataset for Object Detection”. Labelling format of this dataset is same as above mention dataset.
•	Size:
–	Our dataset has total 39,115 images.
–	Each of all 20 activities can found in 3000 different images.

•	Format of dataset label:
o	Required Format:
–	{Class Id }  { X Center }  { Y Center }  { Width }  { Height }
–	Ex. 7   0.549165   0.654981   0.845461   0.456554

o	Other Dataset Format:
–	Pascal Format:
{ Frame Id }  { X Min }  { Y Min }  { X Max }  { Y Max }
Ex. 000001 623 172 768 594


SYSTEM DOCUMENTATION

–	Custom Format: 
{ Person Id }  { X Min }  { Y Min }  { X Max }  { Y Max }  
{ Frame Id }  { Lost }  { Occluded }  { Generated }  “person” 
{ +actions }
Ex.  6  1402  522  1489  610  250  0  0 0  "Person"  "Walking“  "Hugging“

•	Dataset Split:
We split dataset into two main parts training data and validation data,

 

–	The train data is the portion of the dataset that is used to train the machine learning model. The goal of train data is to learn a function that maps inputs to outputs, based on the patterns observed in the train data.

–	The validation data, on the other hand, is used to evaluate the performance of the model during training. This is done by using the validation set to test the model on data it has not seen before, and comparing its predictions to the actual target values. This helps to estimate how well the model will generalize to new, unseen data.



SYSTEM DOCUMENTATION

	Training Parameters: -
•	--data-path = Provide path of dataset yaml file.
•	--conf-file = Provide experiments description file.
•	--img-size = Set image size while training.
•	--batch-size = Set training batch number.
•	--epochs = Set epoch number for training.
•	--device = Set CPU or index of GPU.
•	--eval-interval = Set evaluation interval rate.
•	--output-dir = Path to save output.
•	--name = Provide name of sub-folder in which all weight files save.
•	--save_ckpt_on_last_n_epoch = Set number to save last n number of weight files.


	Inference Parameters: -
•	--weights =  Path of  model weight file
•	--source =  Path to test images and videos
•	--webcam = Set inference on camera other wise use source
•	--webcam-addr = Provide webcam address
•	--yaml = Provide path of dataset yaml file
•	--img-size = Set image size while training.
•	--device = Set CPU or index of GPU.
•	--save-dir = Path to save output.
       
SYSTEM DOCUMENTATION
    
   
   
   
SYSTEM DOCUMENTATION

	Confusion matrix: -
 
	Live Camera Inference: - 
There are several methods to do Live Camera Inferencing, such as Desktop Application, Mobile Application and Website.

–	We create Website to do so. As it is less time consuming and easy to rich wide range of users.
–	For Website we use Flask backend framework.
–	But it is on development stage.

SYSTEM DOCUMENTATION
   


**REFERENCES AND BIBLIOGRAPHY**

	References: -
–	https://github.com/meituan/YOLOv6 
–	https://www.google.com/
–	https://www.youtube.com/
–	https://colab.research.google.com/ 
–	http://okutama-action.org/ 
–	https://vision.cs.uiuc.edu/projects/activity/ 
–	https://www.kaggle.com/datasets/f445f341fc5e3ab58757efa983a38d6dc709de82abd1444c8817785ecd42a1ac 
–	https://learnopencv.com/yolov6-object-detection/ 


	Bibliography: -
–	From GitHub we get YOLOv6 repository.
–	Google and YouTube we use to learn and do some research.
–	In Colab we train our model.
–	From Okutama, vision and Kaggle we got dataset.
–	From Learnopencv.com we get information about YOLOv6


