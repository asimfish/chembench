#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Example - Getting tracker positions

This example shows how to list and read tracker data
"""

import sys
sys.path.insert(0, '..')

from tracker import SteamVRTrackerSDK


def main():
    print("=" * 60)
    print("SteamVR Tracker SDK - Basic Example")
    print("=" * 60)
    print()
    
    # Create SDK instance
    sdk = SteamVRTrackerSDK()
    
    # Initialize connection
    print("[1/3] Initializing connection to SteamVR...")
    if not sdk.initialize():
        print("[ERROR] Failed to initialize. Make sure SteamVR is running.")
        return
    
    print("[SUCCESS] Connected to SteamVR")
    print()
    
    # List all trackers
    print("[2/3] Listing available trackers...")
    trackers = sdk.list_trackers()
    
    if not trackers:
        print("[WARN] No trackers found!")
        print("Make sure:")
        print("  - Trackers are powered on")
        print("  - Trackers are paired with SteamVR")
        print("  - Base stations are active")
        sdk.shutdown()
        return
    
    print(f"[INFO] Found {len(trackers)} tracker(s):")
    for i, tracker in enumerate(trackers, 1):
        print(f"  [{i}] Serial: {tracker['serial']}")
        print(f"      Model: {tracker['model']}")
        print(f"      Index: {tracker['index']}")
        
        # Get battery level if available
        battery = sdk.get_tracker_battery(tracker['serial'])
        if battery is not None:
            print(f"      Battery: {battery*100:.1f}%")
    
    print()
    
    # Get pose data
    print("[3/3] Reading tracker poses...")
    poses = sdk.get_all_trackers()
    while(True):
        for serial, pose in poses.items():
            # print(f"\nTracker: {serial}")
            # print(f"  Position (m):")
            print(f"    X: {pose.x:+.4f}, Y: {pose.y:+.4f}, Z: {pose.z:+.4f}")
            # print(f"    ")
            # print(f"    ")
            
            # print(f"  Orientation (quaternion):")
            # print(f"    X: {pose.qx:+.4f}, Y: {pose.qy:+.4f}, Z: {pose.qz:+.4f}, W: {pose.qw:+.4f}")
            # print(f"    ")
            # print(f"    ")
            # print(f"    ")
            
            # print(f"  Orientation (Euler degrees):")
            # print(f"    Roll:  {np.degrees(pose.roll):+.2f}°")
            # print(f"    Pitch: {np.degrees(pose.pitch):+.2f}°")
            # print(f"    Yaw:   {np.degrees(pose.yaw):+.2f}°")
            
            # print(f"  Velocity (m/s):")
            # print(f"    VX: {pose.vx:+.4f}")
            # print(f"    VY: {pose.vy:+.4f}")
            # print(f"    VZ: {pose.vz:+.4f}")
            
            # print(f"  Status: {pose.tracking_status.name}")
        
    # Cleanup
    print()
    print("[INFO] Shutting down...")
    sdk.shutdown()
    print("[SUCCESS] Done!")


if __name__ == '__main__':
    import numpy as np
    main()

