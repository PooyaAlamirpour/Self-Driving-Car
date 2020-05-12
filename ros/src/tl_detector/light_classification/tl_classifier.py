from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import glob
#from styx_msgs.msg import TrafficLight
import collections
import tensorflow as tf

from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm

# Import everything needed to edit/save/watch video clips
#import imageio
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#import IPython.display as display
#plt.style.use('ggplot')

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense
import os
#import pathlib


class TLClassifier(object):
    """
    This class implements a very simple traffic light classifier.
    The classifier looks at a picture and counts the pixels in a specific color range.
    To be effective, the colorspace is HSV; here, red and yellow can be distinguished
    with ease. Green traffic lights are neglected because these can be passed.
    """
    def __init__(self):
        """
        This member function initializes the classifier.
        It sets the bounds for image classification and intializes the
        state of a possible traffic light in an image.
        """
        self.image = None
        # Lower bound for color in image to be "valid red" in HSV-color-space (!)
        self.HSV_bound_red_low = np.array([0, 120, 120],np.uint8)
        # Upper bound for color in image to be "valid red" in HSV-color-space (!)
        self.HSV_bound_red_high = np.array([10, 255, 255],np.uint8)
        # Lower bound for color in image to be "valid yellow" in HSV-color-space (!)
        self.HSV_bound_yellow_low = np.array([25, 120, 120],np.uint8)
        # Upper bound for color in image to be "valid yellow" in HSV-color-space (!)
        self.HSV_bound_yellow_high = np.array([45.0, 255, 255],np.uint8)
        # Constant defining how many pixels of certain color must
        # be present to be detected as a valid red or yellow
        # traffic light
        self.number_of_pixels_tolerance = 60
        # Member variable indicating a red traffic light
        self.red_light = False
        # Member variable indicating a red yellow traffic light
        self.yellow_light = False

        # Training parmeters
        self.BATCH_SIZE = 32
        self.IMG_HEIGHT = 224
        self.IMG_WIDTH = 224
        self.CLASS_NAMES = None
        self.EPOCHS = 5
        self.VALIDATION_STEPS=8
        self.trained_model = None
        self.get_labels("light_classification/train/")
        # Frozen inference graph files. NOTE: change the path to where you saved the models.
        #self.SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
        #self.RFCN_GRAPH_FILE = 'rfcn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
        #self.FASTER_RCNN_GRAPH_FILE = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'
        # Colors (one for each class)
        cmap = ImageColor.colormap
        #print("Number of colors =", len(cmap))
        self.COLOR_LIST = sorted([c for c in cmap.keys()])

        #self.detection_graph = self.load_graph(self.SSD_GRAPH_FILE)
        # detection_graph = load_graph(RFCN_GRAPH_FILE)
        # detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        #self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        #self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        #self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        #self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

    def get_labels(self, train_dir="data/"):
        '''
        Get labels from folder structure
        '''
        #train_dir = pathlib.Path(train_dir)
        #print(train_dir)
        #print(glob.glob(train_dir+"*"))

        self.image_count = len(list(glob.glob(train_dir+'*/*.png')))
        print("image_count", self.image_count)
        #print(glob.glob(train_dir+'*'))
        # Get classes/labels from folder structure
        self.CLASS_NAMES = np.array([item.split('/')[-1] for item in glob.glob(train_dir+'*') if item.split('/')[-1] != "LICENSE.txt"])

        print("CLASS_NAMES", self.CLASS_NAMES)


    def crop_generator(self, batches):
        """Take as input a Keras ImageGen (Iterator) and generate random
        crops from the image batches generated by the original iterator.
        """
        while True:
            batch_x, batch_y = next(batches)
            batch_crops = np.zeros((batch_x.shape[0], int(self.IMG_HEIGHT*0.5), batch_x.shape[2], 3))
            for i in range(batch_x.shape[0]):
                batch_crops[i] = img[int(self.IMG_HEIGHT*0.25):int(self.IMG_HEIGHT*0.75), 0:self.IMG_WIDTH, :]
                #batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))

            yield (batch_crops, batch_y)

    def preprocess_image_train(self, img):
        print(img)
        return img[:,:,0] # r,g,b

    def load_datasets(self, train_dir, valid_dir):
        '''
        Loads the training and validation dataset
        '''

        # Folder structure needs to be train/<class/label>/*.png
        self.get_labels(train_dir)


        # Create train generator
        train_datagen = ImageDataGenerator(rescale=1./255,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True)



        # Create validation generator
        valid_datagen = ImageDataGenerator(rescale=1./255)

        self.STEPS_PER_EPOCH = np.ceil(self.image_count/self.BATCH_SIZE)

        train_data_gen = train_datagen.flow_from_directory(directory=str(train_dir),
                                                     batch_size=self.BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                     classes = list(self.CLASS_NAMES))

        #train_data_gen_ = crop_generator(train_data_gen, 224)

        valid_data_gen = valid_datagen.flow_from_directory(directory=valid_dir,
                                                     batch_size=self.BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                     classes = list(self.CLASS_NAMES))

        return train_data_gen, valid_data_gen

    def pipeline(self, mode="train", pred_img_name=None, train_dir="data/", valid_dir="valid/"):
        '''
        Pipeline function that can be used either for training or for usage/prediction
        mode can be one of:
        - train
        - predict
        '''

        if mode == "train":
            train_data_gen, valid_data_gen = self.load_datasets(train_dir, valid_dir)
            model = self.create_model()
            model = self.retrain_weights(model, train_data_gen, valid_data_gen)
            self.save_model(model, "tl_classifier_mobilenet.h5")
        elif mode == "predict":
            pred_img = self.preprocess_image(filename=pred_img_name)
            return self.predict(pred_img)
        else:
            raise ValueError("Invalid mode. Valid modes are either 'train' or 'predict'")

    def preprocess_image(self, filename="", img=None):
        '''
        Loads a single image with either a filename or a loaded image already
        '''
        if filename:
            #img = cv2.imread(filename)
            img = image.load_img(filename, target_size=(self.IMG_WIDTH, self.IMG_HEIGHT))
            img = np.array(img).astype('float32')/255
            img = img[int(self.IMG_HEIGHT*0.25):int(self.IMG_HEIGHT*0.75), 0:self.IMG_WIDTH]
            img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #cv2.imwrite("test.png", img*255)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
            img = np.array(img).astype('float32')/255


        #img = np.array(img).astype('float32')/255
        #img[:,:,1] = 0
        #img[:,:,0] = 0
        img = np.expand_dims(img, axis=0) #[224,224,3] --> [1,224, 224, 3] = (BATCHSIZE, HIGHT, WIDHT, CHANNEL)

        return img


    def retrain_weights(self, model, train_data_gen, valid_data_gen):
        '''
        Re-Trains a part (usually a few of the last layers) of the given model on the
        basis of the given datasets (training and validation)
        '''

        # Print architecture
        for i,layer in enumerate(model.layers):
            print(i,layer.name)
        print(len(model.layers)-3)
        for i in range(len(model.layers)-5):
#             layer.trainable=False
            model.layers[i].trainable = False
        # or if we want to set the first 20 layers of the network to be non-trainable
        # for layer in model.layers[:20]:
        #     layer.trainable=False
        # for layer in model.layers[20:]:
        #     layer.trainable=True
        model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
        # Adam optimizer
        # loss function will be categorical cross entropy
        # evaluation metric will be accuracy

        #step_size_train=train_data_gen.n//train_data_gen.batch_size
        model.fit_generator(generator=train_data_gen,
                           steps_per_epoch=self.STEPS_PER_EPOCH,
                           epochs=self.EPOCHS,
                           validation_data=valid_data_gen,
                           validation_steps=self.VALIDATION_STEPS)
        return model


    def save_model(self, model, filename):
        '''
        Saves the given model to disk
        '''
        model.save(filename)
        print("model saved")

    def load_model(self, filename):
        '''
        Loads a model
        '''
        self.trained_model = self.create_model()
        self.trained_model.load_weights(filename)

    def create_model(self):
        '''
        Creates the model on the basis of a mobile net
        '''
        base_model=MobileNet(input_shape=(self.IMG_HEIGHT,self.IMG_WIDTH,3), weights='imagenet', include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

        x=base_model.output
        x1=GlobalAveragePooling2D()(x)
        x2=Dense(1024,activation='relu')(x1) # we add dense layers so that the model can learn more complex functions and classify for better results.
        x3=Dense(512,activation='relu')(x2) # dense layer 3
        preds=Dense(len(self.CLASS_NAMES), activation='softmax')(x3) # final layer with softmax activation

        model=Model(inputs=base_model.input,outputs=preds)

        return model

    def predict(self, image):
        '''
        Predicts the class of the given image
        '''
        prob_vect = self.trained_model.predict(np.array(image))
        return self.CLASS_NAMES[np.argmax(prob_vect, axis=-1)]

    def get_classification(self, image):
        """
        This member function determines the color of the traffic
        light in the image. It requires an image as input.
        It returns the state of a traffic light as an enumerted type.
        """
        self.red_light = False
        self.yellow_light = False
        self.image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        in_red_range_frame = cv2.inRange(self.image, self.HSV_bound_red_low, self.HSV_bound_red_high)
        number_of_red_pixels = cv2.countNonZero(in_red_range_frame)
        if number_of_red_pixels > self.number_of_pixels_tolerance:
            self.red_light = True
            self.yellow_light = False

        in_yellow_range_frame = cv2.inRange(self.image, self.HSV_bound_yellow_low, self.HSV_bound_yellow_high)
        number_of_yellow_pixels = cv2.countNonZero(in_yellow_range_frame)
        if number_of_yellow_pixels > self.number_of_pixels_tolerance:
            self.red_light = False
            self.yellow_light = True

        if self.red_light:
            return TrafficLight.RED

        if self.yellow_light:
            return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN


    # def vanilla_conv_block(self, x, kernel_size, output_channels):
    #     """
    #     Vanilla Conv -> Batch Norm -> ReLU
    #     """
    #     x = tf.layers.conv2d(
    #         x, output_channels, kernel_size, (2, 2), padding='SAME')
    #     x = tf.layers.batch_normalization(x)
    #     return tf.nn.relu(x)
    #
    # def mobilenet_conv_block(self, x, kernel_size, output_channels):
    #     """
    #     Depthwise Conv -> Batch Norm -> ReLU -> Pointwise Conv -> Batch Norm -> ReLU
    #     """
    #     # assumes BHWC format
    #     input_channel_dim = x.get_shape().as_list()[-1]
    #     W = tf.Variable(tf.truncated_normal((kernel_size, kernel_size, input_channel_dim, 1)))
    #
    #     # depthwise conv
    #     x = tf.nn.depthwise_conv2d(x, W, (1, 2, 2, 1), padding='SAME')
    #     x = tf.layers.batch_normalization(x)
    #     x = tf.nn.relu(x)
    #
    #     # pointwise conv
    #     x = tf.layers.conv2d(x, output_channels, (1, 1), padding='SAME')
    #     x = tf.layers.batch_normalization(x)
    #
    #     return tf.nn.relu(x)


    #
    # Utility funcs
    #
    # def filter_boxes(self, min_score, boxes, scores, classes):
    #     """Return boxes with a confidence >= `min_score`"""
    #     n = len(classes)
    #     idxs = []
    #     for i in range(n):
    #         if scores[i] >= min_score:
    #             idxs.append(i)
    #
    #     filtered_boxes = boxes[idxs, ...]
    #     filtered_scores = scores[idxs, ...]
    #     filtered_classes = classes[idxs, ...]
    #     return filtered_boxes, filtered_scores, filtered_classes
    #
    # def to_image_coords(self, boxes, height, width):
    #     """
    #     The original box coordinate output is normalized, i.e [0, 1].
    #
    #     This converts it back to the original coordinate based on the image
    #     size.
    #     """
    #     box_coords = np.zeros_like(boxes)
    #     box_coords[:, 0] = boxes[:, 0] * height
    #     box_coords[:, 1] = boxes[:, 1] * width
    #     box_coords[:, 2] = boxes[:, 2] * height
    #     box_coords[:, 3] = boxes[:, 3] * width
    #
    #     return box_coords
    #
    # def draw_boxes(self, image, boxes, classes, thickness=4):
    #     """Draw bounding boxes on the image"""
    #     #image = Image.fromarray(image)
    #     ###
    #     # For reversing the operation:
    #     # im_np = np.asarray(im_pil)
    #     ###
    #     draw = ImageDraw.Draw(image)
    #     for i in range(len(boxes)):
    #         bot, left, top, right = boxes[i, ...]
    #         class_id = int(classes[i])
    #         color = self.COLOR_LIST[class_id]
    #         draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

            #image = cv2.rectangle(image, start_point, end_point, color, thickness)
            #cv2.rectangle(image, (left, top), (right, bot), (128,128,128), 2)

    # def load_graph(self, graph_file):
    #     """Loads a frozen inference graph"""
    #     graph = tf.Graph()
    #     with graph.as_default():
    #         od_graph_def = tf.GraphDef()
    #         with tf.gfile.GFile(graph_file, 'rb') as fid:
    #             serialized_graph = fid.read()
    #             od_graph_def.ParseFromString(serialized_graph)
    #             tf.import_graph_def(od_graph_def, name='')
    #     return graph
    #
    # def detect(self, path_to_image):
    #
    #     # Load a sample image.
    #     #image = Image.open(path_to_image)
    #     image = cv2.imread(path_to_image)
    #     #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     #print(image)
    #     #image_np = np.asarray(image, dtype=np.uint8)
    #     image_np = np.asarray(image)
    #     print("After asarray")
    #     print(image_np.shape)
    #     image_np = np.expand_dims(image_np, 0)
    #
    #     with tf.Session(graph=self.detection_graph) as sess:
    #         # Actual detection.
    #         (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
    #                                             feed_dict={self.image_tensor: image_np})
    #
    #         # Remove unnecessary dimensions
    #         boxes = np.squeeze(boxes)
    #         scores = np.squeeze(scores)
    #         classes = np.squeeze(classes)
    #
    #         confidence_cutoff = 0.8
    #         # Filter boxes with a confidence score less than `confidence_cutoff`
    #         boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
    #
    #         # The current box coordinates are normalized to a range between 0 and 1.
    #         # This converts the coordinates actual location on the image.
    #         height = image_np.shape[0]
    #         width = image_np.shape[1]
    #         box_coords = self.to_image_coords(boxes, height, width)
    #         print("Coordinates", box_coords)
    #         print("classes", classes)
    #         # Each class with be represented by a differently colored box
    #         self.draw_boxes(image, box_coords, classes)

            #plt.figure(figsize=(12, 8))
            #plt.imshow(image)


    # def get_labels(self, filename="labels.csv", image_folder="data/*"):
    #     labels=[]
    #     f = open(filename, "w")
    #
    #     for name in glob.glob(image_folder):
    #         labels.append(self.get_classification(cv2.imread(name)))
    #         f.write(str(labels[-1])+",\n")
    #     print(labels)
    #     counter = collections.Counter(labels)
    #     print(counter)
    #     print("UNKNOWN=4, GREEN=2, YELLOW=1, RED=0")
    #
    #
    # def pipeline(self,img):
    #     draw_img = Image.fromarray(img)
    #     boxes, scores, classes = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: np.expand_dims(img, 0)})
    #     # Remove unnecessary dimensions
    #     boxes = np.squeeze(boxes)
    #     scores = np.squeeze(scores)
    #     classes = np.squeeze(classes)
    #
    #     confidence_cutoff = 0.8
    #     # Filter boxes with a confidence score less than `confidence_cutoff`
    #     boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
    #
    #     # The current box coordinates are normalized to a range between 0 and 1.
    #     # This converts the coordinates actual location on the image.
    #     width, height = draw_img.size
    #     box_coords = self.to_image_coords(boxes, height, width)
    #
    #     # Each class with be represented by a differently colored box
    #     self.draw_boxes(draw_img, box_coords, classes)
    #     return np.array(draw_img)


if __name__ == '__main__':

    cls = TLClassifier()

    #for name in glob.glob("data/left0025.jpg"):
    # image = cls.detect("data/left0025.jpg")
    # #print(image)
    # plt.imshow(image)
    # cv2.imwrite("sample1_boxes.jpg", image)
    # plt.show()


    #### Video
    # clip = VideoFileClip('driving.mp4')
    #
    # with tf.Session(graph=cls.detection_graph) as sess:
    #     image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    #     detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    #     detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    #     detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
    #
    #     new_clip = clip.fl_image(cls.pipeline)
    #
    #     # write to file
    #     new_clip.write_videofile('result.mp4')
    cls.pipeline(mode="train", train_dir="train/", valid_dir="valid/")
#     cls.load_model("tl_classifier_mobilenet.h5")
#     print(cls.pipeline(mode="predict", pred_img_name="test/red_traffic_light.png"))
    #print(cls.pipeline(mode="predict", pred_img_name="test/yellow_traffic_light.png"))
    #print(cls.pipeline(mode="predict", pred_img_name="test/green_traffic_light.png"))
#     cls.get_labels("train/")
    #print(cls.CLASS_NAMES)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    # labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    # for image, label in labeled_ds.take(1):
    #     print("Image shape: ", image.numpy().shape)
    #     print("Label: ", label.numpy())
    #display.display(Image.open(str(image_path)))
