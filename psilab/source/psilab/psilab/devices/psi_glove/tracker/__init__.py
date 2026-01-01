
# from teleop_base import TeleOperateDeviceBase
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SteamVR Tracker SDK
===================

A Python SDK for accessing SteamVR tracker position and orientation data.

Basic usage:
    from steamvr_tracker_sdk import SteamVRTrackerSDK
    
    sdk = SteamVRTrackerSDK()
    sdk.initialize()
    
    # Get all trackers
    trackers = sdk.get_all_trackers()
    for serial, pose in trackers.items():
        print(f"{serial}: x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}")
    
    sdk.shutdown()

Context manager usage:
    from steamvr_tracker_sdk import SteamVRTrackerSDK
    
    with SteamVRTrackerSDK() as sdk:
        trackers = sdk.get_all_trackers()
        # ... use trackers ...

Quick functions:
    from steamvr_tracker_sdk import quick_get_trackers, quick_list_trackers
    
    trackers = quick_get_trackers()  # One-shot read
    tracker_list = quick_list_trackers()  # List available trackers
"""

__version__ = '1.0.0'
__author__ = 'SteamVR Tracker Visualization Team'
__license__ = 'MIT'

from .tracker import (
    SteamVRTrackerSDK,
    TrackerPose,
    TrackingStatus,
    quick_get_trackers,
    quick_list_trackers
)

__all__ = [
    'SteamVRTrackerSDK',
    'TrackerPose',
    'TrackingStatus',
    'quick_get_trackers',
    'quick_list_trackers'
]

