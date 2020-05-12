from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

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
