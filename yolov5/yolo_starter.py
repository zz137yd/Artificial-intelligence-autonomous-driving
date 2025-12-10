#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Starter (Master program adapted for blocking callbacks)
Purpose: To continuously publish the /YOLO topic and ensure that the line_trace callback of the main control program is triggered.
"""
import rospy
from std_msgs.msg import String
import time

def yolo_publisher():
    # 1. Initialize node
    rospy.init_node('yolo_starter', anonymous=True)

    # 2. Create Publisher
    # queue_size=10
    pub = rospy.Publisher('/YOLO', String, queue_size=10)
    
    # 3. Set the sending frequency
    # Set to 10Hz (10 times per second) for high-frequency transmission
    # Ensure it can be included during the initialization intervals of the main control program.
    rate = rospy.Rate(10)

    print("="*50)
    print("[INFO] YOLO starter is ready.")
    print("[INFO] Sending start signals continuously to /YOLO topic...")
    print("[INFO] Please ensure the main control program (yolo-ros-cnn_tmp1.py) running")
    print("="*50)

    # 4. Continuous sending loop
    while not rospy.is_shutdown():
        msg = "start"
        pub.publish(msg)
        
        # Print sending status to avoid screen refresh being too fast; print once every 10 times.
        # if rospy.get_time() % 1 < 0.1:
        #     print(f"Sending signal: {msg}")
            
        rate.sleep()

if __name__ == '__main__':
    try:
        yolo_publisher()
    except rospy.ROSInterruptException:
        pass