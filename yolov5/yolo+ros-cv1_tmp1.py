# -*- coding: utf-8 -*-
"""
================================================================================
자율주행 RC-CAR 프로젝트 템플릿
================================================================================
과목: 인공지능과 자율주행
목표: RC-CAR 기반 자율주행차를 실내 트랙에서 완주

[학생 구현 항목]
□ process_frame_for_webcam() - 차선 중심 추출
□ line_trace() - YOLO 객체 검출 + 차선 추종 + 제어 명령
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
import numpy as np
import cv2
import math
import threading
import time
import sys
import atexit
import signal

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from serial import Serial
from collections import deque

# YOLO 관련 임포트
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device

# ============================================================================
# 2. 전역 변수 설정
# ============================================================================
obstacle_state = 3          # 장애물 상태 (0: 매우근접, 1: 근접, 2: 주의, 3: 안전)
center_x = 160              # 차선 중심 x좌표 (초기값: 화면 중앙)
strval_now = 90             # 현재 조향각 (90 = 직진)
spdval = 95                 # 기본 속도값 (90~100 범위)
spdval_lock = threading.Lock()  # 속도 변수 동기화용 Lock

# 시리얼 통신 초기화 (아두이노와 통신)
ser = Serial('/dev/arduino', 115200, timeout=1)

# ============================================================================
# 3. 유틸리티 함수 (수정 불필요)
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
        print("[INFO] 안전하게 종료되었습니다.")
    except:
        pass

atexit.register(safe_shutdown)
signal.signal(signal.SIGINT, lambda s, f: (safe_shutdown(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (safe_shutdown(), sys.exit(0)))


# ============================================================================
# 4. [구현 필요] process_frame_for_webcam() - 차선 중심 추출
# ============================================================================

def process_frame_for_webcam(frame):
    """
    웹캠 프레임에서 차선의 중심점을 추출하는 함수
    
    [입력]
    - frame: 웹캠에서 캡처한 BGR 이미지 (numpy array)
    
    [출력]
    - processed_frame: 시각화가 추가된 프레임 (numpy array)
    
    [구현 가이드]
    1. 프레임 크기 조정 (320x180 권장)
    2. 그레이스케일 변환
    3. 하단 ROI(Region of Interest) 영역 설정
    4. 이진화(threshold)로 차선 검출
    5. 검출된 차선의 좌/우 끝점 찾기
    6. 좌/우 끝점의 중앙값 계산 → center_x에 저장
    7. 시각화 (선택사항)
    
    [힌트]
    - cv2.resize(), cv2.cvtColor(), cv2.threshold() 활용
    - np.nonzero()로 흰색 픽셀 위치 찾기
    - center_x는 전역 변수로 선언되어 있음 (global center_x)
    """
    global center_x
    
    # ========== 여기에 코드를 작성하세요 ==========
    
    # 1. 프레임 리사이즈
    # frame = cv2.resize(...)
    
    # 2. 그레이스케일 변환
    # gray = cv2.cvtColor(...)
    
    # 3. ROI 영역 설정 (하단 1/10 영역 권장)
    # height, width = frame.shape[:2]
    # roi_y = ...
    
    # 4. ROI 영역 이진화
    # _, thresholded = cv2.threshold(...)
    
    # 5. 차선 좌/우 끝점 찾기
    # nonzero_x = np.nonzero(thresholded)[1]
    # left_x = ...
    # right_x = ...
    
    # 6. 중심점 계산
    # center_x = (left_x + right_x) // 2
    
    # 7. 시각화 (선택)
    # cv2.circle(frame, (center_x, roi_y), 10, (0, 0, 255), -1)
    
    # =============================================
    
    return frame


# ============================================================================
# 5. [구현 필요] line_trace() - 메인 주행 로직
# ============================================================================

@torch.no_grad()
def line_trace(msg):
    """
    YOLO 객체 검출 + 차선 추종 + 속도/조향 제어를 수행하는 메인 함수
    
    [입력]
    - msg: ROS String 메시지 (트리거 용도)
    
    [구현 가이드]
    1. YOLO 모델 로드 (최초 1회)
    2. 카메라 스트림 초기화
    3. 프레임 루프:
       a. process_frame_for_webcam()으로 차선 중심 추출
       b. 차선 중심 기반 조향각 계산
       c. YOLO로 객체(신호등, 표지판, 보행자 등) 검출
       d. 검출 객체에 따른 속도 조절
       e. 장애물 상태(obstacle_state) 반영
       f. 시리얼로 제어 명령 전송
    
    [조향각 계산 공식]
    angle = math.degrees(math.atan(((160 - center_x) * 0.65) / 140)) * 2.5
    steering = int(90 - angle)  # 90이 직진, <90 좌회전, >90 우회전
    steering = max(45, min(135, steering))  # 범위 제한
    
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
    - '40_sign', '80_sign': 속도제한 표지판 → 속도 조절
    """
    global strval_now, obstacle_state
    
    # ========== 여기에 코드를 작성하세요 ==========
    
    # ----- 1. YOLO 모델 로드 -----
    weights = './yolov5_traffic.pt'
    data = './data/traffic.yaml'
    device = select_device('0')  # GPU 사용
    
    # yolo_model = DetectMultiBackend(weights, device=device, data=data)
    # stride, names = yolo_model.stride, yolo_model.names
    # imgsz = check_img_size((640, 640), s=stride)
    # yolo_model.warmup(imgsz=(1, 3, *imgsz))
    
    # ----- 2. 카메라 스트림 초기화 -----
    # cudnn.benchmark = True
    # dataset = LoadStreams('0', img_size=imgsz, stride=stride, auto=True)
    
    # ----- 3. 프레임 처리 루프 -----
    # for path, im, im0s, vid_cap, s in dataset:
    #     
    #     # 3a. 차선 중심 추출
    #     processed_frame = process_frame_for_webcam(im0s[0])
    #     
    #     # 3b. 조향각 계산
    #     angle = math.degrees(math.atan(((160 - center_x) * 0.65) / 140)) * 2.5
    #     steering = int(90 - angle)
    #     steering = max(45, min(135, steering))
    #     
    #     # 3c. YOLO 추론
    #     im_tensor = torch.from_numpy(im).to(device).float() / 255.0
    #     if len(im_tensor.shape) == 3:
    #         im_tensor = im_tensor[None]
    #     pred = yolo_model(im_tensor)
    #     pred = non_max_suppression(pred, 0.25, 0.45)
    #     
    #     # 3d. 검출 객체별 속도 제어
    #     speed = spdval  # 기본 속도
    #     for det in pred:
    #         if len(det):
    #             for *xyxy, conf, cls in det:
    #                 label = names[int(cls)]
    #                 bbox_area = (xyxy[2]-xyxy[0]) * (xyxy[3]-xyxy[1])
    #                 # 객체별 처리...
    #     
    #     # 3e. 장애물 상태 반영
    #     if obstacle_state == 0:      # 20cm 이하 - 즉시 정지
    #         speed = 90
    #     elif obstacle_state == 1:    # 30cm 이하 - 정지
    #         speed = 90
    #     elif obstacle_state == 2:    # 50cm 이하 - 감속
    #         speed = speed - 1
    #     
    #     # 3f. 제어 명령 전송
    #     speed = max(90, min(100, speed))
    #     ser.write(create_command(steering, speed))
    #     strval_now = steering
    #     
    #     time.sleep(0.02)
    #     if rospy.is_shutdown():
    #         break
    
    # =============================================
    
    pass


# ============================================================================
# 6. [구현 필요] obstacle() - 라이다 데이터 처리
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
    # n = len(data.ranges)
    # if n == 0:
    #     obstacle_state = 3
    #     return
    
    # 2. 각도 배열 계산
    # idx = np.arange(n)
    # angles_rad = data.angle_min + idx * data.angle_increment
    # angles_deg = np.degrees(angles_rad)
    # dists_m = np.array(data.ranges)
    
    # 3. 유효한 데이터만 필터링
    # valid = np.isfinite(dists_m) & (dists_m > 0)
    # angles_deg = angles_deg[valid]
    # dists_cm = dists_m[valid] * 100.0
    
    # 4. 전방 섹터 필터링 (예: 전방 180도 기준 ±30도)
    # FWD_DEG = 180.0
    # SECTOR_WIDTH = 30.0
    # sector = np.abs(angles_deg - FWD_DEG) <= SECTOR_WIDTH
    
    # 5. 최소 거리 계산 및 상태 결정
    # if np.any(sector):
    #     min_dist = np.min(dists_cm[sector])
    #     if min_dist <= CRITICAL_CM:
    #         obstacle_state = 0
    #     elif min_dist <= NEAR_CM:
    #         obstacle_state = 1
    #     elif min_dist <= MID_CM:
    #         obstacle_state = 2
    #     else:
    #         obstacle_state = 3
    # else:
    #     obstacle_state = 3
    
    # =============================================
    
    pass


# ============================================================================
# 7. [구현 필요] laser_listener() - ROS 노드 초기화 및 구독
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
    # rospy.init_node('laser_listener', anonymous=True)
    
    # 2. 토픽 구독
    # rospy.Subscriber('/YOLO', String, line_trace)
    # rospy.Subscriber('/scan', LaserScan, obstacle, queue_size=1)
    
    # 3. 콜백 대기 (블로킹)
    # rospy.spin()
    
    # =============================================
    
    pass


# ============================================================================
# 8. [구현 필요] 메인 블록
# ============================================================================

if __name__ == '__main__':
    """
    프로그램 시작점
    
    [구현 가이드]
    1. laser_listener()를 호출하여 ROS 노드 시작
    
    [선택 사항]
    - 키보드 제어 스레드 추가 가능
    """
    
    # ========== 여기에 코드를 작성하세요 ==========
    
    # laser_listener()
    
    # =============================================
    
    pass

