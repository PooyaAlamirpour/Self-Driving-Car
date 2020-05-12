#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import os
import sys
import math
import numpy as np

# This calibration paramter debounces the light state
# received from the camera, such that toggeling between
# different states is avoided in case the tl_classifier
# is not sure
STATE_COUNT_THRESHOLD = 5
# This calibration paramter decides if images are saved
# to the linux-filesystem. This may sacrifice some computational
# power in favour of having the images for later analysis.
SAVE_CAMERA_IMAGES_IS_ACTIVE = False
# This calibration paramter decides if the traffic classifer
# light classifer is used or the state of the traffic light
# is taken from the simulator. Turn this to True only when
# using the code in the simulator!
USE_TRAFFIC_LIGHT_STATE_FROM_SIMULATOR = False
# This calibration paramter renders the rate for
# proceesing images and detecting traffic lights
# It should be chosen by ansering the question how fast
# do images change and traffic lights disappear?
# Unit is Hz
TRAFFIC_LIGHT_DETECTION_UPDATE_FREQUENCY = 2
# This calibration parameter allwos to tune the threshold in meters for paying
# attention to the state of traffic light. Below that threshold, camea images
# are processed, above this is not done.
SAFE_DISTANCE_TO_TRAFFIC_LIGHT = 60 #80
SAFE_DISTANCE_TO_STOP_LINE = 40 # 60
# Distance to start decelerating. This distance is the threshold in meters for
# starting to slow down the vehicle. This parameter is related with the definition
# of the functions to define the reference velocity, thus, when modifing it, the 
# MID_POINT parameter in waypoint_updater must be modified also. This distance is 
# measured from the 
DISTANCE_START_DECELERATING = 120 # 180

# State machine parameters
NUM_PREV_STATES = 5 # Number of previous states to be saved
CRUISE = 1
DECELERATING = 2
STOPPED = 3
SPEEDING_UP = 4

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.directory_for_images = '/data/'
        self.image_counter = 0
        self.image_counter_red = 0
        self.image_counter_yellow = 0
        self.image_counter_green = 0
        self.tl_prev_states = [-1]*NUM_PREV_STATES # Save the last five states
        self.car_state = CRUISE
        self.current_vel = None
        self.counter_stopped = 0
        
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_line_positions = self.config['stop_line_positions']
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.distance_to_traffic_light_pub = rospy.Publisher('/distance_to_traffic_light', Int32, queue_size=1)
        self.distance_to_stop_line_pub = rospy.Publisher('/distance_to_stop_line', Int32, queue_size=1)
        self.stopped_time_pub = rospy.Publisher('/stopped_time', Int32, queue_size=1)#JUST FOR DEBUGGING
        self.close_to_tl_pub = rospy.Publisher('/close_to_tl', Bool, queue_size=1)
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        try:
            self.light_classifier.load_model("light_classification/tl_classifier_mobilenet.h5")
        except ValueError:
            print("Cannot find classification model. Check if it exists.")
            USE_TRAFFIC_LIGHT_STATE_FROM_SIMULATOR = True
            rospy.loginfo("tl_detector: Since no classification model can be found, set USE_TRAFFIC_LIGHT_STATE_FROM_SIMULATOR to True")

        self.listener = tf.TransformListener()
        
        self.loop() #rospy.spin()

    def loop(self):
        """
        This member function manages all threads inside the tl_detector node and
        makes the execution deterministic.
        """
        rate = rospy.Rate(TRAFFIC_LIGHT_DETECTION_UPDATE_FREQUENCY)
        while not rospy.is_shutdown():
            if not None in (self.waypoints, self.pose, self.camera_image):
                light_wp, state,close_to_tl = self.process_traffic_lights()
                output_light_wp = light_wp
                prev_state = self.car_state
                #rospy.loginfo('light_wp',light_wp,'prev_state',self.state)
                '''
                Publish upcoming red lights at camera frequency.
                Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
                of times till we start using it. Otherwise the previous stable state is
                used.
                '''
                ''' State machine '''
                self.tl_prev_states.pop(0)
                self.tl_prev_states.append(state)
                # Counts the number of red light detections in the last NUM_PREV_STATES
                count_red = self.tl_prev_states.count(TrafficLight.RED)
                # Counts the number of green light detections in the last NUM_PREV_STATES
                count_green = self.tl_prev_states.count(TrafficLight.GREEN)
                
                pub_light_wp = -1
                
                if (self.car_state  == CRUISE) and (light_wp>=0):
                    '''If it is in cruise and count_red is higher than the specified value
                    our car starts decelerating'''
                    self.car_state = DECELERATING
                    self.last_wp = light_wp
                    pub_light_wp = light_wp
                    rospy.loginfo("tl_detector: DECELERATING")
                if (self.car_state  == DECELERATING):
                    pub_light_wp = self.last_wp
                    #if (light_wp==-1):
                    #    self.car_state = SPEEDING_UP
                    #    pub_light_wp = -1
                    #if close_to_tl:
                        #if (count_red>=3):
                        #    ''' If it is decelerating and detects red light, it updates the last_wp
                        #    in order to stop in the stop line'''
                        #    self.last_wp = light_wp
                        #    pub_light_wp = light_wp
                    if (count_green>=5):
                        ''' If it is decelerating but detects green light, it continues in
                        cruise'''
                        self.car_state = SPEEDING_UP
                        pub_light_wp = -1
                        rospy.loginfo("tl_detector: SPEEDING_UP")
                    if (abs(self.current_vel)<=0.5):
                        ''' If it is decelerating and the velocity is lower than specified it
                        goes to stopped state'''
                        self.car_state = STOPPED
                        rospy.loginfo("tl_detector: STOPPED")
                if (self.car_state  == STOPPED):
                    pub_light_wp = self.last_wp
                    stopped_time = self.counter_stopped/TRAFFIC_LIGHT_DETECTION_UPDATE_FREQUENCY
                    self.stopped_time_pub.publish(stopped_time)
                    if (count_green>=5):# or stopped_time>30:
                        ''' If it is stopped and our traffic light turns on green, it changes
                        to speeding up'''
                        self.car_state = SPEEDING_UP
                        pub_light_wp = -1
                        rospy.loginfo("tl_detector: SPEEDING_UP")
                        self.counter_stopped = 0
                    self.counter_stopped = self.counter_stopped + 1
                if (self.car_state  == SPEEDING_UP):
                    pub_light_wp = -1
                    if self.beyond_tl():
                        '''If it is beyond the traffic light, it goes to cruise state'''
                        self.car_state = CRUISE
                        self.tl_prev_states = [-1]*NUM_PREV_STATES
                        rospy.loginfo("tl_detector: CRUISE")
                
                #rospy.loginfo('prev_state %s'%prev_state+' state %s'%self.car_state+' prev_light_wp %s'%output_light_wp+' pub_light_wp %s'%pub_light_wp)
                #rospy.loginfo('light_wp',light_wp,'prev_state',self.state)
                
                self.upcoming_red_light_pub.publish(Int32(pub_light_wp))
                self.close_to_tl_pub.publish(close_to_tl)
                '''        
                if self.state != state:
                    self.state_count = 0
                    self.state = state

                elif self.state_count >= STATE_COUNT_THRESHOLD:
                    self.last_state = self.state
                    light_wp = light_wp if state == TrafficLight.RED else -1
                    self.last_wp = light_wp
                    self.upcoming_red_light_pub.publish(Int32(light_wp))

                else:
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))

                self.state_count += 1

            #else:
            #    rospy.loginfo("tl_detector: Missing information, traffic light detection aborted.")
            '''
            rate.sleep()


    def pose_cb(self, msg):
        """
        This member function is called when pose is published in order to keep
        the current pose as a member variable.
        """
        self.pose = msg


    def waypoints_cb(self, waypoints):
        """
        This member function is called when waypoints is published in order to keep
        the waypoints as a member variable.
        """
        self.waypoints = waypoints
        number_of_waypoints = len(self.waypoints.waypoints)
        #rospy.loginfo("tl_detector: Catched %d waypoints", number_of_waypoints)


    def traffic_cb(self, msg):
        """
        This member function is called when the state of the traffic lights are published in order to keep
        is as a member variable.
        """
        self.lights = msg.lights


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        #rospy.loginfo("tl_detector: Catched an image.")
        self.has_image = True
        self.camera_image = msg
        if SAVE_CAMERA_IMAGES_IS_ACTIVE:
            self.save_image(msg)


    def save_image(self, img):
        """
        This member function catches images and saves them to disc.
        Arguments:
            img: The image from the simulator.
        """
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        img.encoding = "rgb8"
        cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
#         pred_img = self.light_classifier.preprocess_image(img=cv_image)
        pred_img = cv2.resize(cv_image, (224,224))
#         pred_img = np.array(img).astype('float32')/255
#         pred_img = np.expand_dims(img, axis=0)
        file_name = curr_dir + self.directory_for_images+ 'none/img_'+'%06d'% self.image_counter +'.png'
        self.image_counter += 1
        stop_line_waypoint_index = -1
        state_of_traffic_light = TrafficLight.UNKNOWN
        light = None
        stop_line_position = None
        stop_line_waypoint_index = None
        distance = lambda a,b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        if not None in (self.waypoints, self.pose):
            vehicle_index = self.get_index_of_closest_waypoint_to_current_pose(self.pose.pose.position)
            vehicle_position = self.waypoints.waypoints[vehicle_index].pose.pose.position
            traffic_light_index = self.get_index_of_closest_traffic_light_to_current_pose(vehicle_position)

            if traffic_light_index >= 0:
                traffic_light_waypoint_index = self.get_index_of_closest_waypoint_to_current_pose(self.lights[traffic_light_index].pose.pose.position)
                traffic_light_position = self.waypoints.waypoints[traffic_light_waypoint_index].pose.pose.position

                if traffic_light_waypoint_index > vehicle_index:
                    distance_to_traffic_light = distance(vehicle_position, traffic_light_position)
                    
                    if distance_to_traffic_light < SAFE_DISTANCE_TO_TRAFFIC_LIGHT * 2 and distance_to_traffic_light > 15:
                        traffic_light_state = self.lights[traffic_light_index].state
                        if traffic_light_state == TrafficLight.RED:
                            file_name = curr_dir + self.directory_for_images+ 'red/img_'+'%06d'% self.image_counter_red +'.png'
                            self.image_counter_red += 1
                            self.image_counter -= 1
                            cv2.imwrite(file_name, pred_img)
                        elif traffic_light_state == TrafficLight.YELLOW:
                            file_name = curr_dir + self.directory_for_images+ 'yellow/img_'+'%06d'% self.image_counter_yellow +'.png'
                            self.image_counter_yellow += 1
                            self.image_counter -= 1
                            cv2.imwrite(file_name, pred_img)
                        elif traffic_light_state == TrafficLight.GREEN:
                            file_name = curr_dir + self.directory_for_images+ 'green/img_'+'%06d'% self.image_counter_green +'.png'
                            self.image_counter_green += 1
                            self.image_counter -= 1
                            cv2.imwrite(file_name, pred_img)
        if self.image_counter % 4 == 0:
            cv2.imwrite(file_name, pred_img)
            
#         self.image_counter += 1
       # rospy.loginfo("tl_detector.py: Camera image saved to %s!", file_name)


    def get_light_state(self, light):
        """
        This member function determines the current color of the traffic light.
        Arguments:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        # Return the state of the light for testing
        if not USE_TRAFFIC_LIGHT_STATE_FROM_SIMULATOR:

            # Load image from camera to variable
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            # Preprocess image (Normalizing, Cropping, converting to array, expanding in dimensions)
            pred_img = self.light_classifier.preprocess_image(img=cv_image)

            # Call predict function and return classname as string (Therefore the classifier is independent from ROS Message types and can be tested outside the ROS environment)
            classname = self.light_classifier.predict(pred_img)

            # Map the predicted string to a Traffic Light State
            if classname == "red":
#                 rospy.loginfo("tl_detector.py: Red light detected, publishing: %s", str(TrafficLight.RED))
                rospy.loginfo("Red")
                return TrafficLight.RED
            elif classname == "yellow":
#                 rospy.loginfo("tl_detector.py: Yelllow light detected, publishing: %s", str(TrafficLight.YELLOW))
                rospy.loginfo("Yellow")
                return TrafficLight.YELLOW
            elif classname == "green" or classname == "none":
#                 rospy.loginfo("tl_detector.py: Green light detected, publishing: %s", str(TrafficLight.GREEN))
                rospy.loginfo("Green")
                return TrafficLight.GREEN
            else:
#                 rospy.loginfo("tl_detector.py: Unknown class from traffic light classification, publishing: %s", str(TrafficLight.UNKNOWN))
                rospy.loginfo("None")

        else:
            rospy.loginfo("tl_detector.py: Traffic light state taken from simulator!")
            return light.state

        if not self.has_image:
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN


    def process_traffic_lights(self):
        """
        This member function finds the closest visible traffic light, if one
        exists, and determines its location and color.
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        stop_line_waypoint_index = -1
        state_of_traffic_light = TrafficLight.UNKNOWN
        light = None
        stop_line_position = None
        stop_line_waypoint_index = None
        close_to_tl = False
        distance = lambda a,b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        if not None in (self.waypoints, self.pose):
            vehicle_index = self.get_index_of_closest_waypoint_to_current_pose(self.pose.pose.position)
            vehicle_position = self.waypoints.waypoints[vehicle_index].pose.pose.position
            traffic_light_index = self.get_index_of_closest_traffic_light_to_current_pose(vehicle_position)

            if traffic_light_index >= 0:
                traffic_light_waypoint_index = self.get_index_of_closest_waypoint_to_current_pose(self.lights[traffic_light_index].pose.pose.position)
                traffic_light_position = self.waypoints.waypoints[traffic_light_waypoint_index].pose.pose.position
                distance_to_traffic_light = distance(vehicle_position, traffic_light_position)
                self.distance_to_traffic_light_pub.publish(distance_to_traffic_light)
                light = self.lights[traffic_light_index]
                stop_line_index = self.get_index_of_closest_stop_line_to_current_pose(traffic_light_position)
                stop_line_position = self.get_stop_line_positions()[stop_line_index].pose.pose
                stop_line_waypoint_index = self.get_index_of_closest_waypoint_to_current_pose(stop_line_position.position)
                stop_line_waypoint_index -= 10;
                stop_line_wp_position = self.waypoints.waypoints[stop_line_waypoint_index].pose.pose.position
                distance_to_stop_line = distance(vehicle_position, stop_line_wp_position)
                
                if stop_line_waypoint_index>vehicle_index:
                    self.distance_to_traffic_light_pub.publish(distance_to_traffic_light)
                    self.distance_to_stop_line_pub.publish(distance_to_stop_line)
                    if distance_to_stop_line < SAFE_DISTANCE_TO_STOP_LINE:
                        close_to_tl = True
                        state_of_traffic_light = self.get_light_state(light)
                        #rospy.loginfo("tl_detector: Traffic light ahead: {}".format(distance_to_traffic_light))
                        #rospy.loginfo("tl_detector: Traffic light has state: {}".format(state_of_traffic_light))
                        return stop_line_waypoint_index, state_of_traffic_light, close_to_tl
                    elif distance_to_stop_line< DISTANCE_START_DECELERATING:
                        return stop_line_waypoint_index, TrafficLight.UNKNOWN, close_to_tl
                        
#         rospy.loginfo("tl_detector: Stop light detection failed.")
        return -1, TrafficLight.UNKNOWN, close_to_tl

    def get_index_of_closest_waypoint_to_current_pose(self, pose):
        """
        This member functions returns the index of the waypoint that is closest
        to current pose.
        Return:
             an integer, -1 means the search has not been succesfull
        """
        return self.get_index_of_closest_point_to_current_pose(pose, self.waypoints.waypoints)


    def get_index_of_closest_traffic_light_to_current_pose(self, pose):
        """
        This member functions returns the index of the traffic light that is closest
        to current pose.
        Return:
             an integer, -1 means the search has not been succesfull
        """
        return self.get_index_of_closest_point_to_current_pose(pose, self.lights)


    def get_index_of_closest_stop_line_to_current_pose(self, pose):
        """
        This member functions returns the index of the stop line that is closest
        to current pose.
        Return:
             an integer, -1 means the search has not been succesfull
        """
        return self.get_index_of_closest_point_to_current_pose(pose, self.get_stop_line_positions())

    def get_stop_line_positions(self):
        """
        This member function returns a vector-of-vectors of stop-light-positions.
        Returns:
            stop_line_positions: array of 2-d-arrays
        """
        stop_line_positions = []
        for position in self.config['stop_line_positions']:
            current_point = Waypoint()
            current_point.pose.pose.position.x = position[0]
            current_point.pose.pose.position.y = position[1]
            current_point.pose.pose.position.z = 0.0
            stop_line_positions.append(current_point)

        return stop_line_positions

    def get_index_of_closest_point_to_current_pose(self, pose, positions):
        """
        This member function finds the index of that point in the set of points
        positions, which is closest to the pose.
        Arguments:
            pose: pose
            positions: positions given in "waypoints-structure"
        Returns:
            minimum_distance_idx: an integer, -1 means the search has not been succesfull
        """
        # default value for id
        minimum_distance_idx = -1
        # initialize this with a very large value
        minimum_distance_value = sys.float_info.max
        # define lambda-function for euclidan distance
        distance = lambda a,b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        # loop over all waypoints
        if positions is not None:
            for i in range(len(positions)):
                value = distance(pose, positions[i].pose.pose.position)
                if value < minimum_distance_value:
                    minimum_distance_value = value
                    minimum_distance_idx = i

        return minimum_distance_idx
    
    def velocity_cb(self, msg):
        self.current_vel = msg.twist.linear.x
        
    def beyond_tl(self):
        ''' This function returns True if the car is beyond the traffic light, and False otherwise'''
        if not None in (self.waypoints, self.pose):
            vehicle_index = self.get_index_of_closest_waypoint_to_current_pose(self.pose.pose.position)
            vehicle_position = self.waypoints.waypoints[vehicle_index].pose.pose.position
            traffic_light_index = self.get_index_of_closest_traffic_light_to_current_pose(vehicle_position)

            if traffic_light_index >= 0:
                traffic_light_waypoint_index = self.get_index_of_closest_waypoint_to_current_pose(self.lights[traffic_light_index].pose.pose.position)
                return traffic_light_waypoint_index<vehicle_index
            
        return False
                
       
if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
