import sys
import time
import argparse
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
from threadpoolctl import threadpool_limits
import numpy as np
# Added imports for robust config loading
import os
import yaml
from ament_index_python.packages import get_package_share_directory
try:
    from kalman_filter import KalmanFilter6D
    from tracker import ViveTrackerModule
except:
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).parent))
    from kalman_filter import KalmanFilter6D
    from tracker import ViveTrackerModule

# 导入配置读取器（保留兼容，但优先使用本文件的加载方式）
try:
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent / 'config'))
    from config_reader import load_robot_config as legacy_load_robot_config
except Exception:
    legacy_load_robot_config = None

# LHR-23E126B7 right_hand
# LHR-BECE1FC5 left_hand

def get_init_tcp_pose_mat(hand_name):
    tcp_mat = np.eye(4)
    if hand_name == 'right':
        tcp_mat = np.eye(4)
        tcp_mat[:3, 3] = np.array([0.7510337, -0.45611531, 1.16706606])
        tcp_mat[:3, :3] = np.array([
            [ 0.99495139, -0.0925498,  -0.03881072,  
            0.09331946,  0.99546418,  0.01850818,
            0.03692175, -0.02203653,  0.99907516]
        ]).reshape(3,3)
        return tcp_mat
    elif hand_name == 'left':
        tcp_mat = np.eye(4)
        tcp_mat[:3, 3] = np.array([0.74861398, 0.39416502, 1.16217356])
        tcp_mat[:3, :3] = np.array([
            [ 0.99773781, -0.0385833,  -0.05505083,  
            0.04268063,  0.99624673,  0.07530478,
            0.0519387,  -0.07748403,  0.99563979]
        ]).reshape(3,3)
        return tcp_mat
    else:
        raise ValueError(f"Invalid hand name: {hand_name}")


class ViveTrackerNode(Node):
    def __init__(self):
        super().__init__('vive_tracker_node')
        
        # 从配置文件加载默认值
        config = self._load_config()
        
        # 声明参数（使用配置文件中的默认值）
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('base_frame', 'map')
        self.declare_parameter('use_kalman_filter', True)
        
        # 从配置文件获取追踪器序列号和手部名称的默认值

        """ ty del 2025.09.23 
        default_left_serial = config.get('left_hand_tracker_serial', 'LHR-BECE1FC5')
        default_right_serial = config.get('right_hand_tracker_serial', 'LHR-23E126B7')
        """

        # ty add 2025.09.23 
        self.load_configuration()
        default_left_serial = self.left_hand_tracker_serial
        default_right_serial = self.right_hand_tracker_serial
        # ty add end

        # 根据节点名称或参数确定是左手还是右手
        node_name = self.get_name()
        if 'left' in node_name:
            default_serial = default_left_serial
            default_hand = 'left'
        elif 'right' in node_name:
            default_serial = default_right_serial  
            default_hand = 'right'
        else:
            # 如果无法从节点名称判断，使用右手作为默认
            default_serial = default_right_serial
            default_hand = 'right'
        
        self.declare_parameter('serial_number', default_serial)
        self.declare_parameter('hand_name', default_hand)
        
        self.get_logger().info(f"✅ 从配置文件加载默认值: {default_hand}手追踪器={default_serial}")
        
        # 获取参数
        self.publish_rate = self.get_parameter('publish_rate').value        
        self.publish_tf = self.get_parameter('publish_tf').value
        self.base_frame = self.get_parameter('base_frame').value
        self.use_kalman_filter = self.get_parameter('use_kalman_filter').value
        self.hand_name = self.get_parameter('hand_name').value
        self.serial_number = self.get_parameter('serial_number').value
        self.device_key = 'tracker'
        
        # 初始化Vive Tracker模块
        self.vive_tracker_module = ViveTrackerModule()
        self.vive_tracker_module.print_discovered_objects()

        # 初始化6D Pose卡尔曼滤波器
        # self.kf = KalmanFilter6D(dt=1/self.publish_rate)
        
        # 获取所有tracker设备
        all_tracking_devices = self.vive_tracker_module.return_selected_devices(self.device_key)
        
        # 根据序列号重新映射设备
        self.tracking_device = self._get_device_by_serial(all_tracking_devices)
        self.get_logger().info(f"✅ 序列号 {self.serial_number} 的设备为 {self.hand_name}")

        # 初始化ROS2发布器
        self.pose_pub = self.create_publisher(PoseStamped, f'/vive_tracker/{self.hand_name}/pose', 10)
        self.tcp_sub = self.create_subscription(PoseStamped, f'/mink_fk/{self.hand_name}_tcp_pose', self.tcp_state_callback, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # 根据手套类型标志位设置旋转校准矩阵
        glove_type_flag = self._get_glove_type_flag(config)
        base_angles = [180/180*np.pi, 0, 90/180*np.pi]
        calibrated_angles = [angle * glove_type_flag for angle in base_angles]
        
        self.rot_cali_mat = np.eye(4)
        self.rot_cali_mat[:3, :3] = R.from_euler('xyz', calibrated_angles).as_matrix()
        
        self.get_logger().info(f"✅ {self.hand_name}手手套类型标志位: {glove_type_flag}")
        self.get_logger().info(f"✅ 校准角度: {[f'{angle*180/np.pi:.1f}°' for angle in calibrated_angles]}")
        self.world_rot_cali_mat = np.eye(4)
        self.world_rot_cali_mat[:3, :3] = R.from_euler('xyz', [90/180*np.pi, 0, 180/180*np.pi]).as_matrix()

        self.tracker2wrist_mat = np.eye(4)
        self.tracker2wrist_mat[:3, 3] = np.array([0.0, 0.0, -0.11])

        self.init_tcp_pose_mat = get_init_tcp_pose_mat(self.hand_name)  # the predefined init tcp pose of the real robot      
        self.init_tracker_pose = self.calibrate_tracker_system(self.tracking_device.get_T().copy())
        self.init_tracker_pos = self.init_tracker_pose[:3, 3]
        self.prev_position = self.init_tcp_pose_mat[:3, 3] # the first position of the tracker, since the tracker always start from zero, so we add zero upon the init robot tcp pose

        # 创建定时器，以指定频率更新tracker数据
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
    
    def tcp_state_callback(self, msg: PoseStamped):
        # self.get_logger().info(f'Received TCP state: {msg}')
        # get tcp psoe mat
        _tcp_pose_mat = np.eye(4)
        _tcp_pose_mat[:3, 3] = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        _tcp_pose_mat[:3, :3] = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]).as_matrix()
        self.current_tcp_state_mat = _tcp_pose_mat
        # self.get_logger().info(f'TCP pose mat: {self.current_tcp_state_mat}')
        
    
    def calibrate_tracker_system(self, T):
        T = self.world_rot_cali_mat @ T @ self.rot_cali_mat @ self.tracker2wrist_mat
        # self.world_rot_cali_mat 矫正tracker的坐标系(这时的人体姿态人站在机器人右侧， 面朝方向跟机器人一直， 手臂前身)，使其与机器人坐标系一致（x面向机器人前方， Z top）
        # self.rot_cali_mat 将tracker自身坐标系与机器人坐标系对齐
        # self.tracker2wrist_mat 因为tracker在手腕上方，所以需要将位置向下移动0.11m
        return T

    def _get_device_by_serial(self, all_devices):
        # 建立序列号到设备的映射
        serial_to_device = {}
        for device_name, device in all_devices.items():
            serial_to_device[device.get_serial()] = device
        
        # 根据配置重新分配设备名称
        device = serial_to_device[self.serial_number]
        self.get_logger().info(f"✅ 将序列号 {self.serial_number} 的设备为 {self.hand_name}")
        return device

    def timer_callback(self):
        """定时器回调函数，更新并发布tracker数据"""
        current_time = self.get_clock().now()
        pose_msg = PoseStamped()

        with threadpool_limits(limits=1, user_api='blas'):
            # 获取tracker的变换矩阵
            T = self.tracking_device.get_T().copy()
            T = self.calibrate_tracker_system(T) # 矫正tracker的坐标系(这时的人体姿态人站在机器人右侧， 面朝方向跟机器人一直， 手臂前身)，使其与机器人坐标系一致（x面向机器人前方， Z top）
            delta_T = np.linalg.inv(self.init_tracker_pose) @ T

            robot_tcp_target_pose = self.init_tcp_pose_mat @ delta_T
            # robot_tcp_target_pose = self.init_tcp_pose_mat

            # set the tracker pose based on the robot init pose
            T = robot_tcp_target_pose
            
            # 提取位置和旋转
            position = T[:3, 3]
            rotation_matrix = T[:3, :3]
            quaternion = R.from_matrix(rotation_matrix).as_quat()

            # 检查位置是否跳跃过大
            _change = np.linalg.norm(position - self.prev_position)
            if _change > 0.1:
                self.get_logger().warn(f"position jump: {_change} m")
                return  # 如果位置跳跃过大，则不发布数据
            self.prev_position = position      

            pose_msg.header = Header()
            pose_msg.header.stamp = current_time.to_msg()
            pose_msg.header.frame_id = self.base_frame
            pose_msg.pose.position.x = float(position[0])
            pose_msg.pose.position.y = float(position[1])
            pose_msg.pose.position.z = float(position[2])
            pose_msg.pose.orientation.x = float(quaternion[0])
            pose_msg.pose.orientation.y = float(quaternion[1])
            pose_msg.pose.orientation.z = float(quaternion[2])
            pose_msg.pose.orientation.w = float(quaternion[3])
        
        # 发布PoseStamped消息
        self.pose_pub.publish(pose_msg)
        
        # 发布TF变换
        if self.publish_tf:
            transform_msg = TransformStamped()
            transform_msg.header = Header()
            transform_msg.header.stamp = current_time.to_msg()
            transform_msg.header.frame_id = self.base_frame
            transform_msg.child_frame_id = f'vive_{self.hand_name}'
            
            transform_msg.transform.translation.x = float(position[0])
            transform_msg.transform.translation.y = float(position[1])
            transform_msg.transform.translation.z = float(position[2])
            
            transform_msg.transform.rotation.x = float(quaternion[0])
            transform_msg.transform.rotation.y = float(quaternion[1])
            transform_msg.transform.rotation.z = float(quaternion[2])
            transform_msg.transform.rotation.w = float(quaternion[3])
            
            self.tf_broadcaster.sendTransform(transform_msg)
    
    def _load_config(self):
        """加载 robot_config.yaml（优先已安装的 haptic_hand_control/share/config，其次源码 a2d-tele/config）。"""
        # 首选：已安装的 haptic_hand_control 包中的 config
        candidates = []
        try:
            share_dir = get_package_share_directory('haptic_hand_control')
            candidates.append(os.path.join(share_dir, 'config', 'robot_config.yaml'))
        except Exception:
            pass
        # 兼容：源码树路径
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', '..', 'config', 'robot_config.yaml')))
        # 旧实现：通过 config_reader
        if legacy_load_robot_config is not None:
            try:
                data = legacy_load_robot_config()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        # 逐个尝试候选路径
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        if isinstance(data, dict):
                            return data
                except Exception:
                    continue
        # 兜底默认值
        return {
            'left_hand_tracker_serial': 'LHR-BECE1FC5',
            'right_hand_tracker_serial': 'LHR-23E126B7',
            'glove_type_flag': {
                'left': -1,   # 默认为air版本
                'right': -1   # 默认为air版本
            }
        }
    

 #ty add 2025.09.23  增加参数的读取功能
    def load_configuration(self):
        """
        加载配置（优先级系统）：
        1. 首先尝试从 /var/psi/configuration/launch_params.yaml 加载
        2. 如果未找到，从工作空间 config/robot_config.yaml 加载
        3. 自动复制工作空间配置到校准目录
        """
        # 定义配置文件路径
        configuration_dir = '/var/psi/configuration'
        configuration_config_path = os.path.join(configuration_dir, 'launch_params.yaml')
              
        # 首先尝试从校准目录加载配置
        if os.path.exists(configuration_config_path):
            try:
                self.get_logger().info(f'Loading configuration from: {configuration_config_path}')
                with open(configuration_config_path, 'r') as file:
                    config_data = yaml.safe_load(file)
                    self.apply_config_from_yaml(config_data)
                    config_loaded = True
                    self.get_logger().info('Successfully loaded configuration from configuration directory')
            except Exception as e:
                self.get_logger().error(f'Failed to load configuration config: {e}')     
    
        
    def apply_config_from_yaml(self, config_data):
        """
        从YAML结构中提取相机配置参数 
        """
        try:
            # 从YAML结构中提取左手配置参数
            if 'left_hand_tracker_serial' in config_data:
                self.left_hand_tracker_serial = config_data['left_hand_tracker_serial']
            # 从YAML结构中提取右手配置参数
            if 'right_hand_tracker_serial' in config_data:
                self.right_hand_tracker_serial = config_data['right_hand_tracker_serial']
        except Exception as e:
            self.get_logger().error(f'Failed to apply YAML configuration: {e}')

#ty add end


    def _get_glove_type_flag(self, config):
        """获取当前手的手套类型标志位"""
        try:
            glove_type_flags = config.get('glove_type_flag', {})
            flag = glove_type_flags.get(self.hand_name, -1)  # 默认为air版本(-1)
            
            # 验证标志位有效性
            if flag not in [-1, 1]:
                self.get_logger().warn(f"⚠️ 无效的手套类型标志位 {flag}，使用默认值 -1 (air版本)")
                flag = -1
                
            return flag
        except Exception as e:
            self.get_logger().warn(f"⚠️ 读取手套类型标志位失败: {e}，使用默认值 -1 (air版本)")
            return -1

def main(args=None):
    rclpy.init(args=args)
    
    # 创建ViveTrackerNode实例
    node = ViveTrackerNode()
    
    node.get_logger().info(f"开始发布数据...")
    node.get_logger().info("按Ctrl+C停止节点")
    
    # 运行节点
    rclpy.spin(node)
    

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
