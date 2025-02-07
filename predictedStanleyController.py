import rclpy
import math
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import csv

f = open('/home/tushar_rao/Tush_eufs/eufs/controller_ws/src/controller/controller/coords.csv', 'w')
writer = csv.writer(f)
class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/cmd', 10)
        self.get_logger().info('CmdPublisher node started.')
        self.i= -1
        self.k = 0.6
        self.a = 0
        self.j=0
        self.flag = 0
        self.reference_velocity = 9
        self.time_step = 0.11
        self.previous_error  =0.0
        self.L = [
            [4.71199331e+00, 1.01268348e+00],
            [6.68569800e+00, 1.35587473e+00],
            [8.80066596e+00, 1.47857760e+00],
            [1.10227051e+01, 1.47596826e+00],
            [1.33176233e+01, 1.44322289e+00],
            [1.56512284e+01, 1.47551765e+00],
            [1.79889310e+01, 1.66297768e+00],
            [2.02878411e+01, 1.99018810e+00],
            [2.24972075e+01, 2.34178417e+00],
            [2.45659699e+01, 2.59846916e+00],
            [2.64430679e+01, 2.64094631e+00],
            [2.80774411e+01, 2.34991891e+00],
            [2.94200601e+01, 1.61772496e+00],
            [3.04441574e+01, 4.64238094e-01],
            [3.11367137e+01, -1.01190969e+00],
            [3.14849123e+01, -2.71092331e+00],
            [3.14759368e+01, -4.53300767e+00],
            [3.10969706e+01, -6.37836769e+00],
            [3.03384109e+01, -8.15206738e+00],
            [2.92322155e+01, -9.82200855e+00],
            [2.78392950e+01, -1.13998684e+01],
            [2.62211483e+01, -1.28982137e+01],
            [2.44392743e+01, -1.43296114e+01],
            [2.25551722e+01, -1.57066284e+01],
            [2.06269157e+01, -1.70395495e+01],
            [1.86785967e+01, -1.83160180e+01],
            [1.67148687e+01, -1.95107259e+01],
            [1.47401565e+01, -2.05982129e+01],
            [1.27588850e+01, -2.15530188e+01],
            [1.07754790e+01, -2.23496831e+01],
            [8.79447809e+00, -2.29627384e+01],
            [6.83284361e+00, -2.33659348e+01],
            [4.93092121e+00, -2.35315503e+01],
            [3.13164257e+00, -2.34316997e+01],
            [1.47793940e+00, -2.30384981e+01],
            [1.27434006e-02, -2.23240602e+01],
            [-1.22134705e+00, -2.12611463e+01],
            [-2.19962297e+00, -1.98571434e+01],
            [-2.92395429e+00, -1.81708850e+01],
            [-3.39836801e+00, -1.62653799e+01],
            [-3.62689114e+00, -1.42036367e+01],
            [-3.61355069e+00, -1.20486642e+01],
            [-3.36237367e+00, -9.86347121e+00],
            [-2.87738708e+00, -7.71106645e+00],
            [-2.16261794e+00, -5.65445867e+00],
            [-1.22209325e+00, -3.75665662e+00],
            [-5.98400245e-02, -2.08066905e+00],
            [1.32011474e+00, -6.89504699e-01],
            [2.91374403e+00, 3.53827675e-01]
        ]
    
        self.subscription = self.create_subscription(
            Odometry,
            '/ground_truth/odom',
            self.odom_callback,
            10
        )
    def odom_callback(self, msg):
        orientation = msg.pose.pose.orientation
        yaw = self.quaternion_to_euler(orientation.x, orientation.y, orientation.z, orientation.w)

        velocity = msg.twist.twist.linear
        vel_x = velocity.x
        vel_y = velocity.y

        w1 = self.L[self.i%len(self.L)]  # waypoint coords
        w2 = self.L[(self.i+1)%len(self.L)]

        position = msg.pose.pose.position
        x_pos, y_pos = position.x, position.y
        way1 = self.to_car_frame(w1, (x_pos, y_pos), yaw)
        way2 = self.to_car_frame(w2, (x_pos, y_pos), yaw)     
        t = self.check_waypoint_crossed((x_pos, y_pos), w1, w2)
        print(self.i)

        # Check if the robot has crossed the waypoint
        heading_error = self.compute_heading_error(way1, way2)  # Negative when car heading left of bearing
        #yaw = (yaw+np.pi)%(2*np.pi)-np.pi
        yaw = self.normalize_angle(yaw)
        # heading_error-=yaw
        #heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        # heading_error = self.normalize_angle(heading_error)
        crosstrack_error = self.compute_crosstrack_error((0, 0), way1, way2)  # Positive when car right of path

        car_velocity = self.get_velocity(vel_x, vel_y)

        delta = self.stanley_control_law(heading_error, crosstrack_error, car_velocity, self.k)
        desired_acceleration = self.pidlongitudinal(car_velocity, self.reference_velocity)

        predicted_yaws = self.get_predicted_yaw_list(car_velocity, yaw, delta)
        predicted_positions= self.get_predicted_pos_list(x_pos, y_pos, car_velocity, yaw, predicted_yaws,delta, desired_acceleration)
        error_list = self.get_error_list( predicted_positions, predicted_yaws)
        predicted_delta_list= self.get_predicted_delta_list(error_list, vel_x, vel_y, desired_acceleration)
        predicted_delta = self.get_predicted_delta(predicted_delta_list, delta)
        if t:
            self.i+= 1
            #print("I crossed", w2)
            # print("waypoints: ", (w1,w2))
            # print("predicted positions", predicted_positions)
            # print("car_position", (x_pos, y_pos))
            # print(predicted_delta_list)
            # print("delta: ", delta)
            # if (self.i%len(self.L)) + 2 == len(self.L)-1:
            #     (self.i%len(self.L)) = -1
            #     self.k=0.2
        self.j+=1
        if self.j%30==0:
            written_list = [x_pos, y_pos]
            for i in predicted_positions:
                written_list.append(i[0])
                written_list.append(i[1])
            writer.writerow(written_list)
        if car_velocity>0.75*self.reference_velocity and car_velocity<1.25*self.reference_velocity:
            self.flag = 1

        if self.flag==1:
            delta = predicted_delta
        if delta>1:
            # print("predicted positions", predicted_positions)
            # print("car_position", (x_pos, y_pos))
            # print(predicted_delta_list)
            # print("delta: ", delta)
            delta = 0.0

        elif delta<-1:
            # print("predicted positions", predicted_positions)
            # print("car_position", (x_pos, y_pos))
            # print(predicted_delta_list)
            # print("delta: ", delta)
            delta =0.0


        

        #print("delta: ", delta)
        # print("delta list:", predicted_delta_list)
        # print("predicted delta", predicted_delta)

        # print("yaw: ", predicted_yaws)
        pub_msg = AckermannDriveStamped()
        pub_msg.drive.acceleration = desired_acceleration
        pub_msg.drive.steering_angle = delta
        self.publisher_.publish(pub_msg)

    

    
    def pidlongitudinal(self, car_velocity, reference_velocity):
        # Define PID gains
        Kp = 1.0  # Proportional gain
        Ki = 0.0  # Integral gain
        Kd = 0.1 # Derivative gain


        integral_error = 0.0  # Integral term
 # Error from the previous step
        dt = 0.1  # Time step

        # Compute error
        error = reference_velocity - car_velocity

        # Compute integral
        integral_error += error * dt

        # Compute derivative
        derivative_error = (error - self.previous_error) / dt

        # Compute desired acceleration
        desired_accelaration = Kp * error + Ki * integral_error + Kd * derivative_error

        # Update previous error
        self.previous_error = error

        return desired_accelaration

    def PID(self, car_velocity, t, kp, kd, ki):
        e = self.reference_velocity - car_velocity 
        accel = kp*e
        return accel
    def stanley_control_law(self, heading_error, crosstrack_error, car_velocity, k):
        if car_velocity!=0:
            k_dampened=k/(1+0.035*abs(car_velocity))
            delta = (float(heading_error) - np.arctan((k_dampened*crosstrack_error)/car_velocity))
        else:
            delta = 0.0
        return delta
    
    def to_car_frame(self, global_point, car_position, car_orientation):
        translated_x = global_point[0] - car_position[0]
        translated_y = global_point[1] - car_position[1]
        cos_theta = math.cos(-car_orientation)
        sin_theta = math.sin(-car_orientation)
        return translated_x * cos_theta - translated_y * sin_theta, translated_x * sin_theta + translated_y * cos_theta 

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
        return (np.arctan2(dy,dx) + np.pi) % (2* np.pi) - np.pi


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
    def get_predicted_pos_list(self, x_pos, y_pos, car_velocity,yaw,  yaw_list, delta, accel):
        predicted_pos_list = []
        pred_x, pred_y = (x_pos, y_pos)
        yaw_list.insert(0, yaw)
        d = delta
        for i in range(3):
            vel_x=car_velocity*np.cos(yaw_list[i]+d)+accel*self.time_step
            vel_y=car_velocity*np.sin(yaw_list[i]+d)+accel*self.time_step
            pred_x+=vel_x*self.time_step
            pred_y+=vel_y*self.time_step
            closest_p, second_closest_p = [4.71199331e+00, 1.01268348e+00], [4.71199331e+00, 1.01268348e+00]

            closest, second_closest = float('inf'), float('inf')
            for j in self.L[((self.i%len(self.L))+1)%len(self.L):len(self.L):1]:
                dist = self.calc_distance((pred_x, pred_y), j)
                if dist<closest:
                    second_closest= closest
                    second_closest_p= closest_p
                    closest = dist
                    closest_p= j
                elif dist<second_closest:
                    second_closest= dist
                    second_closest_p = j
            
            future_nearest_waypoint= self.L[max(self.L.index(closest_p), self.L.index(second_closest_p))%len(self.L)]
            past_nearest_waypoint = self.L[(self.L.index(future_nearest_waypoint)-1)%len(self.L)]

            c_error = self.compute_crosstrack_error((pred_x, pred_y), past_nearest_waypoint, future_nearest_waypoint)
            h_error = self.compute_heading_error(past_nearest_waypoint, future_nearest_waypoint)
            h_error = self.normalize_angle(h_error-yaw_list[i])
            d = self.stanley_control_law(h_error, c_error, car_velocity, self.k)
            predicted_pos_list.append((pred_x, pred_y))
        return predicted_pos_list
    def get_predicted_yaw_list(self, car_velocity, yaw, delta):
        predicted_yaw_list = [] 
        #predicted_yaw = (yaw+np.pi)%(2*np.pi)-np.pi
        predicted_yaw = self.normalize_angle(yaw)
        for i in range(1, 4):
            predicted_yaw+=(car_velocity*math.tan(delta)*self.time_step)/1.58
            #predicted_yaw = (predicted_yaw+np.pi)%(2*np.pi)-np.pi
            predicted_yaw = self.normalize_angle(predicted_yaw)
            predicted_yaw_list.append(predicted_yaw)
        return predicted_yaw_list
    def get_error_list(self, predicted_positions, predicted_yaws):
        nearest_waypoints = []
        error_list = []

        for pos in predicted_positions:
            closest, second_closest = float("inf"), float("inf")
            closest_p, second_closest_p =[4.71199331e+00, 1.01268348e+00], [4.71199331e+00, 1.01268348e+00]
            for wp in self.L[((self.i%len(self.L))+1)%len(self.L):len(self.L):1]:
                dist = self.calc_distance(wp, pos)
                if dist<closest:
                    second_closest=closest
                    closest=dist
                    second_closest_p= closest_p
                    closest_p = wp
                elif dist<second_closest:
                    second_closest = dist
                    second_closest_p= wp
            nearest_waypoints.append(max(self.L.index(closest_p), self.L.index(second_closest_p)))
            #print(nearest_waypoints)
            h_error = self.compute_heading_error(self.L[max(self.L.index(closest_p), self.L.index(second_closest_p))-1], self.L[max(self.L.index(closest_p), self.L.index(second_closest_p))])
            #h_error = (h_error + np.pi) % (2 * np.pi) - np.pi
            h_error = self.normalize_angle(h_error- predicted_yaws[predicted_positions.index(pos)])
            ct_error= self.compute_crosstrack_error(pos,self.L[max(self.L.index(closest_p), self.L.index(second_closest_p))-1], self.L[max(self.L.index(closest_p), self.L.index(second_closest_p))] )
            error_list.append((h_error, ct_error))
            

        return error_list
    
    def get_predicted_delta_list(self, error_list, vel_x, vel_y, accel):

        predicted_delta_list = []
        for error in error_list:
            vel_x+=accel*self.time_step
            vel_y+=accel*self.time_step
            car_velocity = math.sqrt(vel_x**2+vel_y**2)
            delta = self.stanley_control_law(error[0], error[1], car_velocity, self.k)
            predicted_delta_list.append(delta)
        return predicted_delta_list
    def get_predicted_delta(self, predicted_delta_list, delta):
        weights = [0.6, 0.2, 0.1, 0.1]
        predicted_delta_list.insert(0, delta)
        final_delta = 0
        for i in range(3):
            final_delta+=weights[i]*predicted_delta_list[i]
        return final_delta
    
    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

        
    def calc_distance(self, p1, p2):
        return math.sqrt((p1[1]-p2[1])**2+(p1[0]-p2[0])**2)
def main(args=None):
    rclpy.init(args=args)
    odom_subscriber = OdomSubscriber()
    rclpy.spin(odom_subscriber)
    rclpy.shutdown()


main()
