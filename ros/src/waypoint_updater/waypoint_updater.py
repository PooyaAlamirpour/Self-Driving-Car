#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32, Bool
import numpy as np
import math
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

It is taken into account the position of the traffic light in order to decelerate waypoints.

This node is subscribed to the following topics:
- current_pose
- base_waypoints: publishes a list of all waypoints for the track, so this list includes waypoints
                     both before and after the vehicle
- traffic_waypoint: it is the index of the waypoint for nearest upcoming red light's stop line
And it publishes final_waypoints, which are the list of waypoints to be followed.

There are two parameters that can be tuned:
- LOOKAHEAD_WPS: which defines the number of waypoints that will be published,
- MAX_DECEL: it is the maximum deceleration to be commanded.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = .5
MID_POINT = 60 # Point where velocity is decreased from approaching velocity to zero

class WaypointUpdater(object):
    def __init__(self):
        rospy.loginfo('Initializing my waypoint_updater.')
        rospy.init_node('waypoint_updater')
        
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.base_lane = None
        self.stopline_wp_idx = -1
        self.alpha = self.calc_coef_c2(MID_POINT)/self.calc_coef_c1(MID_POINT)
        self.close_to_tl = False
        
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/close_to_tl', Bool, self.close_to_tl_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add a subscriber for /obstacle_waypoint below

        
        self.loop()


    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree and self.waypoints_2d:
                # Get closest waypoint
                self.publish_waypoints(self.get_closest_waypoint_idx())
                self.check_stopline()
            rate.sleep()

    # Compute the closest waypoint index
    def get_closest_waypoint_idx(self):
        # Get the car's current position
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        # Note: .query returns (distance, index)
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        prev_idx = closest_idx - 1
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[prev_idx]

        # Equation for hyperplane through closest_coords
        closest_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(closest_vect - prev_vect, pos_vect - closest_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    # Publish the main output final_waypoints
    def publish_waypoints(self, closest_id):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    # This function generates the lane which will be sent
    def generate_lane(self):
        lane = Lane()

        # Compute the closest index to our position
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS

        # Slice base_waypoints with our closest and farthest indexes
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx>=farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane
        #pass
    def check_stopline(self):
        # Get stopline coordinates
        sl_x = self.base_waypoints.waypoints[self.stopline_wp_idx].pose.pose.position.x
        sl_y = self.base_waypoints.waypoints[self.stopline_wp_idx].pose.pose.position.y
        # Check if stopline is ahead or behind vehicle
        yaw_rad = self.get_yaw()
        yaw_deg = yaw_rad*180/np.pi
        
        yaw_vect = (np.cos(yaw_rad), np.sin(yaw_rad))
        sl_vect = (sl_x,sl_y)
        pos_vect = np.array(self.get_car_xy())
        
        val = np.dot(yaw_vect, sl_vect - pos_vect)
        close_wp = self.get_closest_waypoint_idx()
        sl_wp = self.stopline_wp_idx
        #print('val: ', val, 'close_wp: ', close_wp, 'sl_wp: ', sl_wp)
        
    def get_yaw(self):
        q = self.pose.pose.orientation;
        q = (q.x, q.y, q.z, q.w)
        #print(self.pose.pose.orientation)
        #rotation = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        attitude = tf.transformations.euler_from_quaternion(q)
        return attitude[2]
    
    def get_car_xy(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        return x,y
    
    # Function to decelerate the waypoints between our position and the traffic light
    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        stop_idx = max(self.stopline_wp_idx - closest_idx -2, 0)
        print('stop_idx', stop_idx)
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            
            dist = self.distance(waypoints, i, stop_idx)
                
            #vel = math.sqrt(2*MAX_DECEL*dist)
            #vel_coef = (dist/20)**2
            #vel_coef = 0.0
            #vel_coef = 1-(2/(1+math.exp(dist/15)))
            
            if self.close_to_tl:
                vel_coef = self.alpha*self.calc_coef_c1(dist)
            else:
                vel_coef = self.calc_coef_c2(dist)
                
            if vel_coef >1:
                vel_coef = 1
            vel = vel_coef* wp.twist.twist.linear.x
            if vel<1.:
                vel = 0.
            p.twist.twist.linear.x = vel
            temp.append(p)

        return temp
    
    def calc_coef_c1 (self,dist):
        return (-(1/(1+math.exp(dist/10)))+0.5)
    
    def calc_coef_c2 (self,dist):
        return (-(0.5/(1+math.exp((dist-(MID_POINT+50))/10)))+1)
    
    # Callback function when receiving current_pose
    def pose_cb(self, msg):
        self.pose = msg

    # Callback function when receiving base_waypoints
    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [
                [w.pose.pose.position.x, w.pose.pose.position.y]
                for w in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    # Callback function when receiving close_to_tl
    def close_to_tl_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.close_to_tl = msg.data
        
    # Callback function when receiving traffic_waypoint
    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data


    # Callback for /obstacle_waypoint message
    def obstacle_cb(self, msg):

        pass


    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x


    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    # Compute distance between two waypoints
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
