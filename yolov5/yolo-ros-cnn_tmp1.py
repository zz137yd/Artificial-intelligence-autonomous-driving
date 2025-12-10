# -*- coding: utf-8 -*-
"""
================================================================================
자율주행 RC-CAR 프로젝트 템플릿 (CNN 버전)
================================================================================
과목: 인공지능과 자율주행
목표: RC-CAR 기반 자율주행차를 실내 트랙에서 완주
특징: 차선 추종에 CNN 모델 사용 (OpenCV 기반 대신)

[학생 구현 항목]
□ process_frame_for_cnn() - CNN 입력 전처리 및 조향각 예측
□ line_trace() - YOLO 객체 검출 + CNN 차선 추종 + 제어 명령
□ obstacle() - 라이다 데이터 처리
□ laser_listener() - ROS 노드 및 구독
□ main 블록 - 스레드 시작
================================================================================
"""

# ============================================================================
# 1. 필요한 라이브러리 임포트
# ============================================================================
import rospy
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import cv2
cv2.setNumThreads(0)
import math
import threading
import time
import sys
import atexit
import signal
from PIL import Image

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from serial import Serial
from collections import deque

# YOLO 관련 임포트
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device

# CNN 조향 모델 임포트
from CNN_Model.CNN_model import SteeringModel

import os
import pathlib

if os.name == 'posix':
    pathlib.WindowsPath = pathlib.PosixPath
# ============================================================================
# 2. 전역 변수 설정
# ============================================================================
obstacle_state = 3          # 장애물 상태 (0: 매우근접, 1: 근접, 2: 주의, 3: 안전)
strval_now = 90             # 현재 조향각 (90 = 직진)
spdval = 98                 # 기본 속도값 (90~100 범위)
spdval_lock = threading.Lock()  # 속도 변수 동기화용 Lock
steering_history = deque(maxlen=3)  # 조향각 평활화용 히스토리

line_trace_running = False      # Whether this function is running

# State machine definition
STATE_DRIVE = 0      # Normal driving
STATE_STOPPED = 1    # Parking

current_state = STATE_DRIVE    # Current status
stop_counter = 0               # A counter that continuously detects stopped objects (debouncing)
clear_counter = 0              # Counter for consecutively no stopped objects detected (recovery confirmation)）

# Threshold parameter
CONFIRM_STOP_FRAMES = 3        # A stop is only implemented after three consecutive frames of detection (to prevent false detections due to flickering)
CONFIRM_RESUME_FRAMES = 17     # Resume only after 17 consecutive frames (approximately 3 seconds) of no detection (to prevent sudden starts due to detection loss)

# 시리얼 통신 초기화 (아두이노와 통신)
ser = Serial('/dev/arduino', 115200, timeout=1)

# ============================================================================
# 3. CNN 입력 전처리 파이프라인
# ============================================================================
# CNN 모델 입력에 맞게 이미지를 변환하는 transforms 정의
transformations = transforms.Compose([
    transforms.Resize((60, 320)),           # 크기 변환 (높이 60, 너비 320)
    transforms.ToTensor(),                   # 텐서 변환 [0,255] → [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3)  # 정규화 → [-1, 1]
])

# ============================================================================
# 4. 유틸리티 함수 (수정 불필요)
# ============================================================================

def create_command(steering, speed):
    """
    RC-CAR 제어 명령 패킷 생성
    - steering: 조향각 (45~135, 90=직진)
    - speed: 속도 (90=정지, 91~100=전진)
    """
    STX, ETX, Length = 0xEA, 0x03, 0x03
    dummy1, dummy2 = 0x00, 0x00
    Checksum = ((~(Length + steering + speed + dummy1 + dummy2)) & 0xFF) + 1
    return bytearray([STX, Length, steering, speed, dummy1, dummy2, Checksum, ETX])


def safe_shutdown():
    """프로그램 종료 시 차량 정지"""
    try:
        ser.write(create_command(90, 90))  # 정지 명령
        time.sleep(0.5)
        ser.close()
        cv2.destroyAllWindows()
        print("[INFO] 안전하게 종료되었습니다.")
    except:
        pass

atexit.register(safe_shutdown)
signal.signal(signal.SIGINT, lambda s, f: (safe_shutdown(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (safe_shutdown(), sys.exit(0)))


# ============================================================================
# 5. [구현 필요] process_frame_for_cnn() - CNN 입력 전처리 및 조향각 예측
# ============================================================================

def process_frame_for_cnn(frame, cnn_model, device):
    """
    웹캠 프레임을 CNN 모델에 입력하여 조향각을 예측하는 함수
    
    [입력]
    - frame: 웹캠에서 캡처한 BGR 이미지 (numpy array)
    - cnn_model: 로드된 CNN 조향 모델 (SteeringModel)
    - device: 연산 장치 (cuda 또는 cpu)
    
    [출력]
    - steering_angle: 예측된 조향각 (int, 45~135 범위)
    
    [구현 가이드]
    1. 프레임 크기 조정 (320x180)
    2. BGR → RGB 변환 후 PIL Image로 변환
    3. 하단 ROI 영역 크롭 (y: 120~180, 즉 하단 60픽셀)
    4. transforms를 적용하여 텐서로 변환
    5. CNN 모델로 조향각 예측
    6. 조향각 범위 제한 (45~135)
    
    [힌트]
    - cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)로 색상 변환
    - Image.fromarray()로 PIL 이미지 생성
    - img.crop((x1, y1, x2, y2))로 ROI 추출
    - transformations(cropped)로 텐서 변환
    - .unsqueeze(0)으로 배치 차원 추가
    - model(tensor).item()으로 스칼라 값 추출
    """
    
    # ========== 여기에 코드를 작성하세요 ==========
    # 1. 프레임 리사이즈 (320x180)
    img = cv2.resize(frame, (320, 180))
    # 2. BGR → RGB 변환 후 PIL Image로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)


    
    # 3. 하단 ROI 크롭 (y: 120~180)
    cropped = img_pil.crop((0, 120, 320, 180))  # (x1, y1, x2, y2)
    
    # 4. 전처리 및 텐서 변환
    tensor = transformations(cropped).unsqueeze(0).to(device)
    
    # 5. CNN 모델로 조향각 예측
    with torch.no_grad():
        angle = cnn_model(tensor).item()
    
    # 6. 조향각 범위 제한
    steering_angle = int(angle)
    steering_angle = max(45, min(135, steering_angle))
    
    return steering_angle


# ============================================================================
# 6. [구현 필요] line_trace() - 메인 주행 로직
# ============================================================================

@torch.no_grad()
def line_trace(msg):
    """
    YOLO 객체 검출 + CNN 차선 추종 + 속도/조향 제어를 수행하는 메인 함수
    
    [입력]
    - msg: ROS String 메시지 (트리거 용도)
    
    [구현 가이드]
    1. YOLO 모델 로드 (최초 1회)
    2. CNN 조향 모델 로드 (최초 1회)
    3. 카메라 스트림 초기화
    4. 프레임 루프:
       a. process_frame_for_cnn()으로 조향각 예측
       b. 조향각 평활화 (이동 평균)
       c. YOLO로 객체(신호등, 표지판, 보행자 등) 검출
       d. 검출 객체에 따른 속도 조절
       e. 장애물 상태(obstacle_state) 반영
       f. 시리얼로 제어 명령 전송
    
    [CNN 모델 설정]
    - 모델 클래스: SteeringModel (from CNN_model import SteeringModel)
    - 가중치 파일: './best_model.pth'
    - 입력 크기: (60, 320) - ROI 크롭 후 크기
    - 출력: 조향각 (연속값)
    
    [YOLO 설정]
    - weights: './yolov5_traffic.pt'
    - data: './data/traffic.yaml'
    - imgsz: (640, 640)
    - conf_thres: 0.25
    - iou_thres: 0.45
    
    [검출 가능 객체 및 권장 동작]
    - 'red': 빨간 신호등 → 근접 시 정지
    - 'green': 녹색 신호등 → 유지
    - 'yellow': 황색 신호등 → 감속
    - 'person': 보행자 → 근접 시 정지
    - 'stop_sign': 정지 표지판 → 정지
    - 'car': 앞차량 → 거리에 따라 감속/정지
    """
    global strval_now, obstacle_state, steering_history, line_trace_running
    global current_state, stop_counter, clear_counter

    # 1. Check the startup signal!
    if msg.data != "start":
        print(f"Received unknown signal: {msg.data}，ignore...")
        return  # If it's not `start`, exit the function directly and don't run the following code
    
    # 2. Prevent duplicate execution
    if line_trace_running:
        # print("[WARN] line_trace is already running, ignoring repetitive signals")
        return

    line_trace_running = True  # ← Marked as running

    print(">>> Start signal received! Begin loading model and driving! <<<")
    
    # ========== 여기에 코드를 작성하세요 ==========
    
    # ----- 1. YOLO 모델 로드 -----
    weights = './runs/train/exp24/weights/best.pt'
    data = './data/traffic.yaml'
    device = select_device('0')  # GPU 사용
    
    yolo_model = DetectMultiBackend(weights, device=device, data=data)
    stride, names = yolo_model.stride, yolo_model.names
    imgsz = check_img_size((640, 640), s=stride)
    yolo_model.warmup(imgsz=(1, 3, *imgsz))
    
    # ----- 2. CNN 조향 모델 로드 -----
    steer_model_path = './best_model.pth'
    cnn_model = SteeringModel().to(device)
    cnn_model.load_state_dict(torch.load(steer_model_path, map_location=device))
    cnn_model.eval()
    
    # ----- 3. 카메라 스트림 초기화 -----
    cudnn.benchmark = True
    dataset = LoadStreams('0', img_size=imgsz, stride=stride, auto=True)
    
    # ----- 4. 프레임 처리 루프 -----
    for path, im, im0s, vid_cap, s in dataset:
    #     
    #     # 4a. CNN으로 조향각 예측
        steering = process_frame_for_cnn(im0s[0], cnn_model, device)
    #     
    #     # 4b. 조향각 평활화 (선택사항)
        steering_history.append(steering)
        steering = int(np.mean(steering_history))
        steering = max(45, min(135, steering))
    #     
    #     # 4c. YOLO 추론
        im_tensor = torch.from_numpy(im).to(device).float() / 255.0
        if len(im_tensor.shape) == 3:
            im_tensor = im_tensor[None]
        pred = yolo_model(im_tensor)
        pred = non_max_suppression(pred, 0.25, 0.45)
    #     
    #     # 4d. 검출 객체별 속도 제어
        speed = spdval  # 기본 속도
        is_stop_object_visible = False
        should_slow_down = False
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    label = names[int(cls)]
                    bbox_area = (xyxy[2]-xyxy[0]) * (xyxy[3]-xyxy[1])
    #               # 객체별 처리...

                    if bbox_area < 500:
                        continue

                    if label in ['stop', 'red', 'person']:
                        print(f"[Visual] The label {label} (Area:{bbox_area:.0f})")
                        is_stop_object_visible = True

                    elif label in ['car', 'yellow']:
                        should_slow_down = True
        
        target_speed = 100
        
        if current_state == STATE_DRIVE:
            
            # 【Driving status】
            if is_stop_object_visible:
                stop_counter += 1
                clear_counter = 0
                print(f">>> Suspected stop signal... {stop_counter}/{CONFIRM_STOP_FRAMES}")
            
                # Switching to parking mode only after 3 consecutive frames are detected.
                if stop_counter >= CONFIRM_STOP_FRAMES:
                    current_state = STATE_STOPPED
                    stop_counter = 0
                    print(f"!!! Switching states -> [STOPPED] Parking !!!")
            else:
                if should_slow_down: 
                    print("Slow down")
                    target_speed = 98
                # No object seen, reset counter.
                stop_counter = 0

        elif current_state == STATE_STOPPED:
            # 【Parking status】
            # In this state, forced parking
            target_speed = 90
        
            if not is_stop_object_visible:
                clear_counter += 1
                stop_counter = 0
                print(f"<<< Clear vision... {clear_counter}/{CONFIRM_RESUME_FRAMES}")
            
                # After seeing nothing for 17 consecutive frames, the camera switched back to driving mode.
                if clear_counter >= CONFIRM_RESUME_FRAMES:
                    current_state = STATE_DRIVE
                    clear_counter = 0
                    print(f"!!! Switching states -> [DRIVE] Resumption of driving !!!")
            else:
                # If you see it again in the middle, reset and clear the counter, and continue parking.
                clear_counter = 0
                print(f"!!! Obstacle still detected, keep parked. !!!")

    #     # 4e. 장애물 상태 반영
        if obstacle_state == 0:    # Very close
            print(f"[LIDAR] Emergency braking!" )
            target_speed = 90
        elif obstacle_state == 1:  # Close
            print(f"[LIDAR] Obstacle avoidance parking!")
            target_speed = 90
        elif obstacle_state == 2:  # Warning range
            # If vision indicates it's safe to run, but radar detects something nearby, slow down slightly.
            if current_state == STATE_DRIVE:
                target_speed = 94
                print(f"[LIDAR] Deceleration and obstacle avoidance")
    #     
    #     # 4f. 제어 명령 전송
        #speed = max(90, min(110, speed))

        speed = int(target_speed)
        print(f"s:{speed}")
        steering = int(steering)

        if current_state == STATE_STOPPED:
            print(f"[DEBUG] The state machine requires a stop, speed={speed}")
    
        # Ensure the scope is safe again
        speed = max(90, min(110, speed))
        steering = max(0, min(255, steering))

        # Printing occurs frequently only when the status changes or the vehicle stops.屏
        if current_state == STATE_STOPPED or obstacle_state < 3:
            print(f"Execute -> speed: {speed}, steering: {steering}")

        ser.write(create_command(steering, speed))
        strval_now = steering
    #    
        # Displaying live feed
        display_frame = im0s[0].copy()
    
        # Draw YOLO detection box
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    label = names[int(cls)]
                    x1, y1, x2, y2 = map(int, xyxy)
                
                    if label == 'red':
                        color = (0, 0, 255)  # red
                    elif label == 'green':
                        color = (0, 255, 0)  # green
                    elif label == 'yellow':
                        color = (0, 255, 255)  # yellow
                    elif label == 'person':
                        color = (255, 0, 0)  # blue
                    else:
                        color = (255, 255, 255)  # white
                
                    # Draw a rectangle
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                    # Draw labels
                    label_text = f"{label} {conf:.2f}"
                    cv2.putText(display_frame, label_text, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
        # Display speed and steering information
        info_text = f"Speed: {speed} | Steering: {steering}"
        cv2.putText(display_frame, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
        # Display obstacle status
        obstacle_text = f"Obstacle: {obstacle_state}"
        cv2.putText(display_frame, obstacle_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
        # Display window
        cv2.imshow("RC-Car View", display_frame)
        cv2.waitKey(1)

        time.sleep(0.02)
        if rospy.is_shutdown():
            break
    
    # =============================================
    
    pass


# ============================================================================
# 7. [구현 필요] obstacle() - 라이다 데이터 처리
# ============================================================================

def obstacle(data):
    """
    라이다(LiDAR) 스캔 데이터를 분석하여 전방 장애물 상태를 갱신하는 함수
    
    [입력]
    - data: sensor_msgs/LaserScan 메시지
      - data.ranges: 거리 측정값 배열 (미터 단위)
      - data.angle_min: 시작 각도 (라디안)
      - data.angle_increment: 각도 증분 (라디안)
    
    [출력]
    - obstacle_state 전역 변수 갱신
      - 0: 매우 근접 (20cm 이하) → 즉시 정지
      - 1: 근접 (30cm 이하) → 정지
      - 2: 주의 (50cm 이하) → 감속
      - 3: 안전 (50cm 초과) → 정상 주행
    
    [구현 가이드]
    1. 스캔 데이터에서 각도 배열 계산
    2. 전방 섹터 정의 (예: 전방 ±30도)
    3. 전방 섹터 내 최소 거리 계산
    4. 거리에 따라 obstacle_state 갱신
    
    [힌트]
    - np.degrees()로 라디안→도 변환
    - data.ranges에 inf, nan이 포함될 수 있으므로 np.isfinite()로 필터링
    - 거리는 미터 단위이므로 cm로 변환 필요 (* 100)
    """
    global obstacle_state
    
    # ========== 여기에 코드를 작성하세요 ==========
    
    # 임계값 설정
    CRITICAL_CM = 20.0   # 매우 근접
    NEAR_CM = 30.0       # 근접
    MID_CM = 50.0        # 주의
    
    # 1. 스캔 데이터 파싱
    n = len(data.ranges)
    if n == 0:
        obstacle_state = 3
        return
    
    # 2. 각도 배열 계산
    idx = np.arange(n)
    angles_rad = data.angle_min + idx * data.angle_increment
    angles_deg = np.degrees(angles_rad)
    dists_m = np.array(data.ranges)
    
    # 3. 유효한 데이터만 필터링
    valid = np.isfinite(dists_m) & (dists_m > 0)
    angles_deg = angles_deg[valid]
    dists_cm = dists_m[valid] * 100.0
    
    # 4. 전방 섹터 필터링 (예: 전방 180도 기준 ±30도)
    FWD_DEG = 180.0
    SECTOR_WIDTH = 30.0
    sector = np.abs(angles_deg - FWD_DEG) <= SECTOR_WIDTH
    
    # 5. 최소 거리 계산 및 상태 결정
    if np.any(sector):
        min_dist = np.min(dists_cm[sector])
        if min_dist <= CRITICAL_CM:
            obstacle_state = 0
        elif min_dist <= NEAR_CM:
            obstacle_state = 1
        elif min_dist <= MID_CM:
            obstacle_state = 2
        else:
            obstacle_state = 3
    else:
        obstacle_state = 3
    
    # =============================================
    
    pass


# ============================================================================
# 8. [구현 필요] laser_listener() - ROS 노드 초기화 및 구독
# ============================================================================

def laser_listener():
    """
    ROS 노드를 초기화하고 토픽을 구독하는 함수
    
    [구현 가이드]
    1. rospy.init_node()로 노드 초기화
    2. rospy.Subscriber()로 토픽 구독:
       - '/YOLO' 토픽 (String) → line_trace 콜백
       - '/scan' 토픽 (LaserScan) → obstacle 콜백
    3. rospy.spin()으로 콜백 대기
    
    [힌트]
    - anonymous=True로 노드 이름 충돌 방지
    - queue_size=1로 최신 데이터만 처리
    """
    
    # ========== 여기에 코드를 작성하세요 ==========
    
    # 1. ROS 노드 초기화
    rospy.init_node('laser_listener', anonymous=True)
    
    # 2. 토픽 구독
    rospy.Subscriber('/YOLO', String, line_trace)
    rospy.Subscriber('/scan', LaserScan, obstacle, queue_size=1)
    
    # 3. 콜백 대기 (블로킹)
    rospy.spin()
    
    # =============================================
    
    pass


# ============================================================================
# 9. [구현 필요] 메인 블록
# ============================================================================

if __name__ == '__main__':
    """
    프로그램 시작점
    
    [구현 가이드]
    1. laser_listener()를 호출하여 ROS 노드 시작
    """
    
    # ========== 여기에 코드를 작성하세요 ==========
    
    laser_listener()
    
    # =============================================
    
    pass
