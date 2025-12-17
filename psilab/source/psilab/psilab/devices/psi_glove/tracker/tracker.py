
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SteamVR Tracker SDK - Python API for accessing VR tracker data

This SDK provides a simple Python interface to access position and orientation
data from SteamVR trackers without requiring ROS2.

Author: SteamVR Tracker Visualization Team
License: MIT
"""

import openvr
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time


class TrackingStatus(Enum):
    """Tracking status for a device"""
    NOT_TRACKING = 0
    TRACKING = 1
    CALIBRATING = 2
    OUT_OF_RANGE = 3
    DISCONNECTED = 4


@dataclass
class TrackerPose:
    """Container for tracker position and orientation data"""
    
    # Identification
    serial: str
    model: str
    index: int
    
    # Position (meters)
    x: float
    y: float
    z: float
    
    # Orientation (quaternion: x, y, z, w)
    qx: float
    qy: float
    qz: float
    qw: float
    
    # Orientation (Euler angles in radians: roll, pitch, yaw)
    roll: float
    pitch: float
    yaw: float
    
    # Velocity (meters/second)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    # Angular velocity (radians/second)
    av_x: float = 0.0
    av_y: float = 0.0
    av_z: float = 0.0
    
    # Status
    tracking_status: TrackingStatus = TrackingStatus.TRACKING
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'serial': self.serial,
            'model': self.model,
            'index': self.index,
            'position': {'x': self.x, 'y': self.y, 'z': self.z},
            'orientation': {
                'quaternion': {'x': self.qx, 'y': self.qy, 'z': self.qz, 'w': self.qw},
                'euler': {'roll': self.roll, 'pitch': self.pitch, 'yaw': self.yaw}
            },
            'velocity': {'x': self.vx, 'y': self.vy, 'z': self.vz},
            'angular_velocity': {'x': self.av_x, 'y': self.av_y, 'z': self.av_z},
            'tracking_status': self.tracking_status.name,
            'timestamp': self.timestamp
        }
    
    def get_transformation_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix"""
        # Create rotation matrix from quaternion
        rotation = Rotation.from_quat([self.qx, self.qy, self.qz, self.qw])
        rot_matrix = rotation.as_matrix()
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = [self.x, self.y, self.z]
        
        return transform


class SteamVRTrackerSDK:
    """
    Main SDK class for accessing SteamVR tracker data
    
    Example usage:
        sdk = SteamVRTrackerSDK()
        sdk.initialize()
        
        trackers = sdk.get_all_trackers()
        for serial, pose in trackers.items():
            print(f"Tracker {serial}: pos=({pose.x}, {pose.y}, {pose.z})")
        
        sdk.shutdown()
    """
    
    def __init__(self):
        """Initialize SDK (does not connect to SteamVR yet)"""
        self.vr_system = None
        self.is_initialized = False
        self._tracker_cache = {}  # Cache tracker information
        self._last_update_time = 0.0
    
    def initialize(self) -> bool:
        """
        Initialize connection to SteamVR
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize OpenVR in utility mode (doesn't require HMD)
            openvr.init(openvr.VRApplication_Utility)
            self.vr_system = openvr.VRSystem()
            self.is_initialized = True
            
            # Discover initial trackers
            self._discover_trackers()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize OpenVR: {e}")
            print("[INFO] Make sure SteamVR is running")
            return False
    
    def shutdown(self):
        """Shutdown OpenVR connection"""
        if self.is_initialized:
            try:
                openvr.shutdown()
                self.is_initialized = False
                self.vr_system = None
                print("[INFO] OpenVR shutdown complete")
            except Exception as e:
                print(f"[WARN] Error during shutdown: {e}")
    
    def _get_device_property_string(self, device_index: int, prop: int) -> str:
        """Get string property from device"""
        try:
            return self.vr_system.getStringTrackedDeviceProperty(device_index, prop)
        except:
            return ""
    
    def _discover_trackers(self):
        """Discover all connected trackers"""
        if not self.is_initialized:
            return
        
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if not self.vr_system.isTrackedDeviceConnected(i):
                continue
            
            device_class = self.vr_system.getTrackedDeviceClass(i)
            
            # Look for generic trackers
            if device_class == openvr.TrackedDeviceClass_GenericTracker:
                serial = self._get_device_property_string(i, openvr.Prop_SerialNumber_String)
                
                if serial and serial not in self._tracker_cache:
                    model = self._get_device_property_string(i, openvr.Prop_ModelNumber_String)
                    self._tracker_cache[serial] = {
                        'index': i,
                        'model': model
                    }
    
    def list_trackers(self) -> List[Dict[str, str]]:
        """
        List all available trackers
        
        Returns:
            List of dictionaries containing tracker info
            [{'serial': 'LHR-XXX', 'model': 'Vive Tracker 3.0', 'index': 1}, ...]
        """
        if not self.is_initialized:
            print("[WARN] SDK not initialized. Call initialize() first.")
            return []
        
        self._discover_trackers()
        
        result = []
        for serial, info in self._tracker_cache.items():
            result.append({
                'serial': serial,
                'model': info['model'],
                'index': info['index']
            })
        
        return result
    
    def get_tracker_pose(self, serial: str) -> Optional[TrackerPose]:
        """
        Get pose data for a specific tracker
        
        Args:
            serial: Tracker serial number (e.g., 'LHR-C8B8E776')
        
        Returns:
            TrackerPose object or None if tracker not found/not tracking
        """
        if not self.is_initialized:
            print("[WARN] SDK not initialized. Call initialize() first.")
            return None
        
        # Ensure tracker is in cache
        if serial not in self._tracker_cache:
            self._discover_trackers()
        
        if serial not in self._tracker_cache:
            return None
        
        tracker_info = self._tracker_cache[serial]
        device_index = tracker_info['index']
        
        # Get poses from OpenVR
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        
        pose_data = poses[device_index]
        
        if not pose_data.bPoseIsValid:
            return TrackerPose(
                serial=serial,
                model=tracker_info['model'],
                index=device_index,
                x=0.0, y=0.0, z=0.0,
                qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                roll=0.0, pitch=0.0, yaw=0.0,
                tracking_status=TrackingStatus.NOT_TRACKING,
                timestamp=time.time()
            )
        
        # Extract transformation matrix
        mat = pose_data.mDeviceToAbsoluteTracking
        
        # Position
        x = mat[0][3]
        y = mat[1][3]
        z = mat[2][3]
        
        # Rotation matrix
        rot_matrix = np.array([
            [mat[0][0], mat[0][1], mat[0][2]],
            [mat[1][0], mat[1][1], mat[1][2]],
            [mat[2][0], mat[2][1], mat[2][2]]
        ])
        
        # Convert to quaternion
        rotation = Rotation.from_matrix(rot_matrix)
        quat = rotation.as_quat()  # [x, y, z, w]
        
        # Convert to Euler angles (in radians)
        euler = rotation.as_euler('xyz', degrees=False)
        
        # Velocity
        vx = pose_data.vVelocity[0]
        vy = pose_data.vVelocity[1]
        vz = pose_data.vVelocity[2]
        
        # Angular velocity
        av_x = pose_data.vAngularVelocity[0]
        av_y = pose_data.vAngularVelocity[1]
        av_z = pose_data.vAngularVelocity[2]
        
        return TrackerPose(
            serial=serial,
            model=tracker_info['model'],
            index=device_index,
            x=x, y=y, z=z,
            qx=quat[0], qy=quat[1], qz=quat[2], qw=quat[3],
            roll=euler[0], pitch=euler[1], yaw=euler[2],
            vx=vx, vy=vy, vz=vz,
            av_x=av_x, av_y=av_y, av_z=av_z,
            tracking_status=TrackingStatus.TRACKING,
            timestamp=time.time()
        )
    
    def get_all_trackers(self) -> Dict[str, TrackerPose]:
        """
        Get pose data for all trackers
        
        Returns:
            Dictionary mapping serial numbers to TrackerPose objects
            {'LHR-XXX': TrackerPose(...), ...}
        """
        if not self.is_initialized:
            print("[WARN] SDK not initialized. Call initialize() first.")
            return {}
        
        self._discover_trackers()
        
        result = {}
        for serial in self._tracker_cache.keys():
            pose = self.get_tracker_pose(serial)
            if pose:
                result[serial] = pose
        
        return result
    
    def get_tracker_battery(self, serial: str) -> Optional[float]:
        """
        Get battery level for a tracker
        
        Args:
            serial: Tracker serial number
        
        Returns:
            Battery level (0.0-1.0) or None if not available
        """
        if not self.is_initialized or serial not in self._tracker_cache:
            return None
        
        try:
            device_index = self._tracker_cache[serial]['index']
            battery = self.vr_system.getFloatTrackedDeviceProperty(
                device_index, openvr.Prop_DeviceBatteryPercentage_Float
            )
            return battery
        except:
            return None
    
    def is_tracker_connected(self, serial: str) -> bool:
        """
        Check if a tracker is connected
        
        Args:
            serial: Tracker serial number
        
        Returns:
            True if connected, False otherwise
        """
        if not self.is_initialized:
            return False
        
        if serial not in self._tracker_cache:
            self._discover_trackers()
        
        if serial not in self._tracker_cache:
            return False
        
        device_index = self._tracker_cache[serial]['index']
        return self.vr_system.isTrackedDeviceConnected(device_index)
    
    def __enter__(self):
        """Context manager support"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.shutdown()
        return False


# Convenience functions for quick access
def quick_get_trackers() -> Dict[str, TrackerPose]:
    """
    Quick function to get all tracker poses (one-shot)
    
    Returns:
        Dictionary of tracker poses
    """
    sdk = SteamVRTrackerSDK()
    if sdk.initialize():
        trackers = sdk.get_all_trackers()
        sdk.shutdown()
        return trackers
    return {}


def quick_list_trackers() -> List[Dict[str, str]]:
    """
    Quick function to list all trackers (one-shot)
    
    Returns:
        List of tracker info dictionaries
    """
    sdk = SteamVRTrackerSDK()
    if sdk.initialize():
        trackers = sdk.list_trackers()
        sdk.shutdown()
        return trackers
    return []

