import rclpy
import math
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from rosgraph_msgs.msg import Clock
import numpy as np
import csv
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
f = open('/home/tushar_rao/Tush_eufs/eufs/controller_ws/src/controller/controller/vel.csv', 'w')
writer = csv.writer(f)
f2 = open('/home/tushar_rao/Tush_eufs/eufs/controller_ws/src/controller/controller/ref_vel.csv', 'w')
writer2 = csv.writer(f2)
class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/cmd', 10)
        self.get_logger().info('CmdPublisher node started.')
        self.i = -1
        self.k = 0.56#066
        self.a = 0
        self.no= 1000
        self.j = 0
        self.Ki = 0.0
        self.Kp = 1.1
        self.Kd = 0.1
        self.error_sum = 0.0
        self.previous_error=0.0
        self.max_velocity = 6
        self.c = 0
        self.L =[
            (0.5199999999999996, -0.26000000000000156),
            (4, 0.5),
            (6.0997, 0.46011999999999986),
            (8.958765, 0.6334599999999995),
            (12.904501, 0.5442049999999998),
            (16.50963, 0.9106499999999986),
            (20.142315, 2.0267799999999987),
            (23.03038, 3.0184499999999996),
            (25.728749999999998, 4.120899999999999),
            (27.0522, 3.8519499999999987), 
            (29.24365, 3.0322499999999994),
            (30.2343, 2.159699999999999),
            (31.7488, 0.11077499999999851),
            (32.04075, -1.3419550000000005),
            (31.85275, -4.607830000000001),
            (30.93425, -7.440205000000001),
            (30.10215, -9.54865),
            (28.42945, -11.919970000000001),
            (26.178150000000002, -13.647795),
            (23.266765, -15.095735000000001),
            (20.758225, -16.539845),
            (16.424795, -18.711165),
            (12.720585, -20.563344999999998),
            (9.160635, -22.593),
            (6.340465, -24.4171),
            (3.4438549999999992, -25.2316),
            (2.0123599999999993, -24.977),
            (-0.24815000000000076, -23.59835),
            (-1.1382499999999993, -22.7739),
            (-3.0364499999999985, -20.334455),
            (-3.387450000000001, -16.96707),
            (-3.5478500000000004, -15.764190000000001),
            (-3.2800499999999992, -12.895075),
            (-2.8646499999999993, -10.3),
            (-2.565950000000001, -7.250210000000001),
            (-2.5943500000000004, -4.270505000000001),
            (-1.4700000000000006, -1.7200000000000006),
            (-0.6999999999999993, -0.6400000000000006),
            (0.5199999999999996, -0.26000000000000156),
            (4, 0.5)
        ]
        self.subscription = self.create_subscription(
            Odometry,
            '/ground_truth/odom',
            self.odom_callback,
            10
        )
        qos_profile = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.clock_subscription = self.create_subscription(
            Clock, 
            '/clock',
            self.clock_callback,
            qos_profile
        )
    def clock_callback(self, msg):
        #print("clock")
        time = msg.clock.sec
        #print(time)
    def odom_callback(self, msg):
        orientation = msg.pose.pose.orientation
        yaw = self.quaternion_to_euler(orientation.x, orientation.y, orientation.z, orientation.w)
        print
        if yaw > np.pi:
            yaw -= np.pi
        elif yaw < (-1 * np.pi):
            yaw += np.pi
        velocity = msg.twist.twist.linear
        vel_x = velocity.x
        vel_y = velocity.y

        w1 = self.L[self.i + 1]
        w2 = self.L[self.i + 2]
        position = msg.pose.pose.position
        x_pos, y_pos = position.x, position.y
        angle = np.arctan2(y_pos, x_pos)
        if angle>np.pi/2:
            angle-= np.pi/2
        elif angle<-np.pi/2:
            angle+=np.pi/2
        #print(yaw-angle)
        new_pos = self.to_car_frame((x_pos, y_pos), (x_pos, y_pos), yaw)
        #print(new_pos)

        t = self.check_waypoint_crossed((x_pos, y_pos), w1, w2)
        car_velocity = self.get_velocity(vel_x, vel_y)
        # Check if the robot has crossed the waypoint
        if t:
            self.i += 1
            if self.i + 2 == len(self.L)-1:
                self.i = -1
                self.k=0.2

            
        heading_error= self.compute_heading_error(w1, w2)  # Negative when car heading left of bearing

        if heading_error > np.pi:
            heading_error -= np.pi
        elif heading_error < (-1 * np.pi):
            heading_error += np.pi

        crosstrack_error = self.compute_crosstrack_error((x_pos, y_pos), w1, w2)  # Positive when car right of path
        self.j += 1
        delta = self.stanley_control_law(heading_error, crosstrack_error, car_velocity, self.k)
        reference_velocity = self.get_reference_velocity(self.max_velocity, heading_error)
        if(self.j%120 == 0):
            writer.writerow((self.c, car_velocity))
            writer2.writerow((self.c, reference_velocity))
            self.c+=1
        #print(reference_velocity, "-------->", car_velocity)
        if(self.j%120 == 0):
            self.update_pid(car_velocity, reference_velocity)

        desired_acceleration = self.pidlongitudinal(car_velocity, reference_velocity, self.Kp, self.Ki, self.Kd)

        pub_msg = AckermannDriveStamped()
        pub_msg.drive.acceleration = desired_acceleration
        pub_msg.drive.steering_angle = delta
        self.publisher_.publish(pub_msg)
    def get_reference_velocity(self, max_velocity, delta):
        return max_velocity*abs(np.cos(delta))**(1/7)
    def update_pid(self, car_velocity, reference_velocity):
        self.error_sum += reference_velocity - car_velocity
        self.Kp += 0.01 * self.error_sum
        self.Kd += 0.001 * self.error_sum 
    def pidlongitudinal(self, car_velocity, reference_velocity, kp, ki, kd):

        integral_error = 0.0  # Integral term
        dt = 0.1 

        error = reference_velocity - car_velocity

        integral_error += error * dt

        derivative_error = (error - self.previous_error) / dt

        desired_accelaration = kp * error + ki * integral_error + kd * derivative_error

        self.previous_error = error

        return desired_accelaration
    



    def PID(self, car_velocity, t, kp, kd, ki):
        e = self.reference_velocity - car_velocity 
        accel = kp*e
        return accel
    def stanley_control_law(self, heading_error, crosstrack_error, car_velocity, k):
        if car_velocity!=0:
            delta = (float(heading_error) - (np.arctan((k*crosstrack_error+0.005)/car_velocity)))
        else:
            delta = 0.0
        return delta
    
    def get_velocity(self, vel_x, vel_y):
        r = math.sqrt(vel_x**2+vel_y**2)
        return r
    def compute_crosstrack_error(self,fa, w1, w2):
        x0, y0 = fa
        x1, y1 = w1
        x2, y2 = w2
        numerator = abs(((y2-y1)*x0)-(x2-x1)*y0+x2*y1-y2*x1)
        denominator = math.sqrt((y2-y1)**2+(x2-x1)**2)
        return numerator/denominator
    
    def to_car_frame(self, global_point, car_position, car_orientation):
        translated_x = global_point[0] - car_position[0]
        translated_y = global_point[1] - car_position[1]
        cos_theta = math.cos(-car_orientation)
        sin_theta = math.sin(-car_orientation)
        return translated_x * cos_theta - translated_y * sin_theta, translated_x * sin_theta + translated_y * cos_theta
    
    def computer_crosstrack_error(self,fa, w1, w2):
        x0, y0 = fa
        x1, y1 = w1
        x2, y2 = w2

        A = (y2 - y1)/(x2-x1)
        B = 1
        C = -1 * y1
        numerator = abs(A*x0 + B*y0 + C)
        denominator = math.sqrt(A*A + B*B) 
        # print(numerator, "   ", denominator)
        return numerator/denominator
    def check_waypoint_crossed(self, fa, w1, w2):
        a = np.array(w1) #PREVIOUS WAYPOINT
        b = np.array(fa) #POSITION
        c = np.array(w2) #CURRENT WAYPOINT

        CA = c-a 
        BA = b-a 

        t = np.dot(BA,CA) / np.dot(CA,CA)

        return t>1
    def compute_heading_error(self, w1, w2):
        dy = w2[1] - w1[1]
        dx = w2[0] - w1[0]
        return np.arctan2(dy,dx)


    def compute_perpendicular_distance(self, xa, ya, w1, w2):
        x1, y1 = w1
        x2, y2 = w2

        # Compute the perpendicular distance to the line
        m = (y2 - y1) / (x2 - x1)
        c = -m * x1 + y1
        d = abs(-m * xa + ya - c) / math.sqrt(m**2 + 1)
        return d

    def quaternion_to_euler(self, x, y, z, w):
        # Conversion from quaternion to Euler yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

def main(args=None):
    rclpy.init(args=args)
    odom_subscriber = OdomSubscriber()
    rclpy.spin(odom_subscriber)
    rclpy.shutdown()

main()
