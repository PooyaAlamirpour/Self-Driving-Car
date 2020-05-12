## Self-Driving Car - Integrated System

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Each humankind has a love in his life, and here it is my love! Carla.

![The target vehicle](imgs/MyLove.jpg)

It is my honor that finally, I have done an amazing course after 6 incredible months in the [Udacity](https://www.udacity.com/). In this project, I have implemented an integrated system that based on that a car is going to detect traffic light, keeps in the road, and tries to control his steering and speed in the the robot operating system (ROS). There is a Virtual Machine which is provided by the Udacity that you can find [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Udacity_VM_Base_V1.0.0.zip).
This implementation has many different parts as bellow: 
* Longitudinal and lateral movement 
* Traffic light status detection
* Path planning

before deep dive into the project, let's look at the Architecture of this project that is depicted below:

![The target vehicle](imgs/ROS.png)

The main structure is based on the Robot Operating System and consists of some main modules such as Perception, Planning, and Control. The perception module consists of two main modules. One is named Obstacle detection and another is Traffic Light Detection. Both of them need the current position and image color as inputs. The output of this module is important and can be used for controlling the vehicle. The planning module has two subparts. One is called Waypoint Loader and another is Waypoint Updater. The main task of this module is, keeping the vehicle on the road. If the car wants to move forward or change lane, it has to use this module for making the decision. The final module is named Control. This module is placed due to controlling the car in the simulation.

### Longitudinal and lateral movement
The controller package contains a module that keeps the vehicle on the road. The control issue is consists of the control of the longitudinal by tuning throttle or brake and the control of the lateral by tuning the steering angle. For this module, there are two files that are named `dbw_node.py` and `twist_controller.py`:
* `dbw_node.py`: ROS-node that runs control of the longitudinal and lateral vehicle and communicates with other modules.
* `twist_controller.py`: The class for the control algorithm.

The longitudinal vehicle is controlled by a PID-control. This means that the controller can operate with the bounded actuator inputs. When the control law exceeds the bounds, the limit is commanded, and at the same time, the value of the integrator state is clamped. The lateral vehicle is controlled by a feedforward control law. 

### Traffic light status detection
This module has two essential subparts: 
Traffic lights detection
Classification of the state of the traffic lights
The discovery of traffic lights is based on knowledge of the position of traffic lights in the world. If the vehicle approaches a traffic light and is sufficiently close, the classifier is invoked and fed with the camera image, which is received frequently. The camera looks at the traffic light several times, and once, it is assumed that the light is red, the controller of the longitudinal brings the vehicle to a full stop. Otherwise, the controller of the longitudinal dynamics keeps the movement. This behavior of this node is made as-deterministic-as-possible by adding a `self.loop()` method. 

### Path planning
The path planning algorithm determines the final waypoints to be followed by the vehicle. Due to calculate the path, my algorithm receives the base waypoints, and it selects those to be required. When a traffic waypoint is received, velocity is decreased gradually from maximum to either zero or its mean value. For defining the reference values, two coupled sigmoid functions are used. With this procedure, a smooth trajectory is generated; thus, the car can decrease its speed in a controlled way.

### Using the project
For using this project, you should consider setting up some tools. For more information, check the related [Udacity repository](https://github.com/udacity/CarND-Capstone).
* Clone the project repository: `git clone git@github.com:PooyaAlamirpour/Self-Driving-Car.git`
* Install python dependencies: 
	* `cd CarND-Capstone`
	* `pip install -r requirements.txt`
* Run the code
	* `cd ros`
	* `catkin_make`
	* `source devel/setup.sh`
	* `roslaunch launch/styx.launch`
* Now it is time to run the simulator.	
