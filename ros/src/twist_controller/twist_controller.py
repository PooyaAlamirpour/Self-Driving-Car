import rospy
from lowpass import LowPassFilter
from pid import PID
from std_msgs.msg import Float32
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH= 0.44704
MIN_SPEED = 0.1  # m/s


# Actuator bounds
MIN_THROTTLE = 0.0
MAX_THROTTLE = 0.2
# Filter tuning constants
TAU = 0.5
TS = 0.02
# PID-Controller tuning parameters
KP = 0.05
KI = 0.002
KD = 0

# As explained in the walk-through, break torque needed to keep the vehicle in place.
TORQUE_TO_KEEP_VEHICLE_STATIONARY = 700  # Nm


class Controller(object):
    def __init__(self,
                 vehicle_mass,
                 fuel_capacity,
                 brake_deadband,
                 decel_limit,
                 accel_limit,
                 wheel_radius,
                 wheel_base,
                 steer_ratio,
                 max_lat_accel,
                 max_steer_angle):
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.prev_brake = None
        
        self.yaw_controller = YawController(
            wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)
        self.throttle_controller = PID(
            KP, KI, KD, MIN_THROTTLE, MAX_THROTTLE)
        self.v_low_pass_filter = LowPassFilter(TAU, TS)
        self.last_time = rospy.get_time()
        
        self.ref_vel_pub = rospy.Publisher('/ref_vel', Float32, queue_size=1)
        self.current_vel_x_pub = rospy.Publisher('/current_vel_x', Float32, queue_size=1)
        self.throttle_pub = rospy.Publisher('/throttle', Float32, queue_size=1)
        self.brake_pub = rospy.Publisher('/brake', Float32, queue_size=1)
        self.error_vel_pub = rospy.Publisher('/error_vel', Float32, queue_size=1)


    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0, 0, 0

        current_time = rospy.get_time()
        current_vel = self.v_low_pass_filter.filt(current_vel)
        error_vel = linear_vel - current_vel
        ref_vel = linear_vel
        self.ref_vel_pub.publish(ref_vel)
        steering = self.yaw_controller.get_steering(
            linear_vel, angular_vel, current_vel)
        throttle = self.throttle_controller.step(
            error=error_vel,
            sample_time=current_time - self.last_time)
        brake = 0
        self.throttle_pub.publish(throttle*10)
        if linear_vel < 1 and abs(current_vel) < 1:
            # The vehicle is stopped.
            throttle = 0
            brake = TORQUE_TO_KEEP_VEHICLE_STATIONARY
        elif throttle < 0.1 and error_vel < 0:
            # Velocity error is negative, so we need to slow down.
            throttle = 0
            decel = max(error_vel, self.decel_limit)
            ref_brake = abs(decel) * self.vehicle_mass * self.wheel_radius
            if self.prev_brake is None:
                self.prev_brake = brake
            #Filter
            brake = 0.2*ref_brake + 0.8*self.prev_brake
            
        self.prev_brake = brake
        self.last_vel = current_vel
        self.last_time = rospy.get_time()
        self.error_vel_pub.publish(error_vel)
        self.current_vel_x_pub.publish(current_vel)
        self.brake_pub.publish(brake/100)
        return throttle, brake, steering
