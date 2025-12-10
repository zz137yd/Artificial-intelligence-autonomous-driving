import argparse  # 명령행 인자 파싱을 위한 표준 라이브러리
import os  # 파일/디렉터리 경로 처리 및 OS 상호작용
import cv2  # OpenCV 컴퓨터 비전 라이브러리
import torch  # PyTorch 딥러닝 프레임워크
import numpy as np  # 수치 연산을 위한 NumPy
from PIL import Image  # 이미지 로드를 위한 Pillow
import torchvision.transforms as transforms  # 이미지 전처리 파이프라인 도구
from glob import glob  # 파일 패턴 검색 유틸리티
from model import SteeringModel  # 사용자 정의 조향각 추정 모델 클래스

# ===================== Constants =====================  # 섹션 구분 주석
ANGLE_MIN, ANGLE_MAX = 45.0, 135.0  # 모델 출력 각도를 물리 한계(45~135도)로 클램프
RESIZE_W, RESIZE_H = 320, 180   # 학습 시 사용한 입력 크기(너비, 높이) 고정값
CROP_Y0 = 120                   # 세로 180 중 하단 60픽셀만 사용할 때의 시작 y 좌표


# ===================== Model =====================  # 모델 로드 섹션 시작
@torch.no_grad()  # 함수 전체에 대해 그래프 추적 비활성화(추론 전용)
def load_model(model_path, device, allow_fallback: bool = False):  # 체크포인트 파일을 로드해 모델 초기화
    model = SteeringModel().to(device)  # 모델 인스턴스를 생성하고 CPU/GPU로 이동

    # 1) 가능하면 안전 모드로 로드 (PyTorch 신버전)  # weights_only 사용 시 보안·안정성 향상
    try:  # 예외 처리 시작
        state = torch.load(model_path, map_location=device, weights_only=True)  # 체크포인트 로드(가중치만 로드 시도)
    except TypeError:  # 구버전 PyTorch에서는 weights_only 인자를 지원하지 않음
        # 2) 구버전(PyTorch)이면 weights_only 인자 미지원 → 일반 로드  # 호환성 유지
        state = torch.load(model_path, map_location=device)  # 일반 방식으로 로드

    # 3) 다양한 포맷 정규화  # state_dict 키가 다른 이름일 수 있으므로 정규화
    if isinstance(state, dict) and "state_dict" in state:  # 일반적인 Lightning/Trainer 형식
        sd = state["state_dict"]  # 내부 state_dict 추출
    elif isinstance(state, dict) and "model_state" in state:  # 커스텀 키 사용 케이스
        sd = state["model_state"]  # model_state 키에서 추출
    else:  # 이미 state_dict 형태이거나 텐서 dict인 경우
        sd = state  # 그대로 사용

    # 4) 로드 시도  # 모델 파라미터를 실제 네트워크에 주입
    try:  # 일치 여부 검증
        model.load_state_dict(sd)  # 기본 strict=True로 키/크기 일치 요구
    except RuntimeError:  # 키 불일치 등으로 실패 시
        if allow_fallback:  # 허용되면 예비 경로로 재시도
            # ★ 신뢰 가능한 로컬 파일에 한해 최후 수단(보안 유의)  # 객체 전체 로드 가능성
            full = torch.load(model_path, map_location=device)  # 전체 객체 로드(weights_only=False)
            if isinstance(full, dict) and "state_dict" in full:  # 또 다른 포맷 대응
                sd = full["state_dict"]  # state_dict 재지정
            elif isinstance(full, dict) and "model_state" in full:  # 대체 키 케이스
                sd = full["model_state"]  # model_state 재지정
            else:  # 그 외 포맷
                sd = full  # 그대로 사용
            model.load_state_dict(sd)  # 재적용 시도
        else:  # 대체 경로 불허 시
            raise  # 예외 전파

    model.eval()  # 드롭아웃/배치정규화 등을 추론 모드로 전환
    return model  # 초기화 완료된 모델 반환


# ===================== Pre/Post Processing =====================  # 전처리/후처리 섹션
def get_transform():  # 입력 이미지를 학습과 동일하게 변환하는 함수
    """학습과 동일한 전처리: Resize(320x180) → 하단 60px Crop → ToTensor/Normalize"""  # 문서화 문자열
    return transforms.Compose([  # 변환들을 직렬로 연결
        transforms.Resize((RESIZE_H, RESIZE_W)),  # 크기 표준화: (높이, 너비) 순서에 유의
        transforms.Lambda(lambda img: img.crop((0, CROP_Y0, RESIZE_W, RESIZE_H))),  # 하단 60픽셀 영역만 사용
        transforms.ToTensor(),  # [0,1] 범위 텐서로 변환(CHW)
        transforms.Normalize([0.5] * 3, [0.5] * 3)  # 채널별 평균/표준편차로 정규화
    ])  # Compose 종료


def clamp_angle(x: float) -> float:  # 각도를 물리 제약 범위로 자르는 함수
    return max(ANGLE_MIN, min(float(x), ANGLE_MAX))  # [ANGLE_MIN, ANGLE_MAX] 범위로 클램프


def draw_overlay(frame_bgr: np.ndarray, angle: float) -> np.ndarray:  # 시각화 오버레이 그리기
    h, w = frame_bgr.shape[:2]  # 프레임의 높이/너비 추출
    # 상단 2/3 흐림 (표시용): 학습과 직접 관련은 없고, 시각화 일관성만 유지  # 설명 주석
    top_end = (2 * h) // 3  # 상단 블러 끝지점 계산
    if top_end > 0:  # 유효한 경우에만 적용
        top = frame_bgr[0:top_end, :]  # 상단 영역 슬라이스
        frame_bgr[0:top_end, :] = cv2.GaussianBlur(top, (15, 15), 0)  # 가우시안 블러 적용

    # 하단 1/3 강조 박스 (학습 하단 60px 비율 고정 ≈ 1/3)  # 시각적 강조용
    cv2.rectangle(frame_bgr, (0, h - h // 3), (w, h), (0, 0, 255), 2)  # 붉은 박스로 ROI 강조

    # 각도 표시  # 텍스트 렌더링
    cv2.putText(  # 화면 좌상단에 추정 각도 표시
        frame_bgr,  # 대상 이미지
        f"Steering Angle: {angle:.2f}",  # 표시 문자열(소수점 2자리)
        (10, 30),  # 좌표(픽셀)
        cv2.FONT_HERSHEY_SIMPLEX,  # 폰트 종류
        1,  # 글자 크기 스케일
        (0, 255, 0),  # 글자 색상(BGR)
        2,  # 두께(px)
        lineType=cv2.LINE_AA,  # 안티앨리어싱 적용
    )  # putText 종료
    return frame_bgr  # 오버레이가 적용된 프레임 반환


# ===================== I/O Helpers =====================  # 입출력 보조 함수 섹션
def try_open_writer(out_path: str, fps: float, frame_size: tuple):  # 비디오 저장 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 용 코덱 식별자
    return cv2.VideoWriter(out_path, fourcc, fps, frame_size)  # VideoWriter 인스턴스 반환


def get_unique_exp_folder(base='inference'):  # 저장 폴더를 중복 없이 생성
    os.makedirs(base, exist_ok=True)  # 상위 폴더 보장
    for i in range(1, 1000):  # exp1~exp999 탐색
        trial = os.path.join(base, f'exp{i}')  # 후보 경로 생성
        if not os.path.exists(trial):  # 미존재하면 사용
            return trial  # 사용 가능한 경로 반환
    return os.path.join(base, 'exp999')  # 모두 사용 중이면 마지막 경로 반환


def open_capture(cam_index: int):  # 웹캠 캡처 객체 생성(지연 최소화 우선)
    """웹캠 지연 완화: GStreamer 우선 시도 → 실패 시 기본 VideoCapture로 폴백"""  # 동작 설명
    # GStreamer 파이프라인 (가능한 환경에서 지연/버퍼 최소화)  # Jetson/리눅스 환경 최적화
    try:  # 파이프라인 생성 시 예외 대비
        dev = f"/dev/video{cam_index}"  # 장치 노드 경로 구성
        gst = (  # GStreamer 파이프라인 문자열
            f"v4l2src device={dev} io-mode=2 ! "  # V4L2 소스(mmap)
            "image/jpeg,framerate=30/1 ! "  # MJPEG로 30fps 캡처
            "jpegdec ! videoconvert ! videoscale ! "  # JPEG 디코드 및 색공간/크기 변환
            f"video/x-raw,width={RESIZE_W*2},height={RESIZE_H*2} ! "  # 약간 크게 스케일(후단에서 다시 처리)
            "appsink drop=true sync=false max-buffers=1"  # 지연 최소화 옵션
        )  # 파이프라인 끝
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)  # GStreamer 백엔드로 열기
        if cap.isOpened():  # 성공 시
            return cap  # 해당 캡처 객체 반환
    except Exception:  # 파이프라인 미지원/오류
        pass  # 폴백 절차 진행

    # 폴백: 기본 V4L2  # 범용 경로
    cap = cv2.VideoCapture(cam_index)  # 디폴트 백엔드로 열기
    if cap.isOpened():  # 열렸다면
        # 가능 시 버퍼 크기 축소  # 지연을 줄이기 위해 버퍼 1로 설정
        try:  # 속성 설정 중 예외 대비
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 내부 버퍼 최소화
        except Exception:  # 일부 드라이버는 미지원
            pass  # 무시 후 진행
    return cap  # 캡처 객체(또는 실패 시 비활성 객체) 반환


# ===================== Inference Paths =====================  # 각 입력 타입별 추론 경로
@torch.no_grad()  # 이미지 단일 파일 추론 시 그래프 비활성화
def infer_image(image_path, model, transform, device, save_dir):  # 정지 이미지 추론 함수
    orig_img = Image.open(image_path).convert("RGB")  # 이미지를 RGB로 로드
    # Resize는 transform 내부에서 수행  # 중복 리사이즈 방지
    input_tensor = transform(orig_img).unsqueeze(0).to(device)  # 배치 차원 추가 후 디바이스로 이동
    angle = clamp_angle(model(input_tensor).item())  # 모델 추론 및 각도 클램프

    # 시각화용  # 원본 크기 기준 오버레이
    img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)  # PIL→NumPy(BGR)
    img_cv = draw_overlay(img_cv, angle)  # 오버레이 적용

    if save_dir:  # 저장 모드인 경우
        os.makedirs(save_dir, exist_ok=True)  # 폴더 생성
        out_path = os.path.join(save_dir, os.path.basename(image_path))  # 출력 파일 경로
        cv2.imwrite(out_path, img_cv)  # 결과 저장
        print(f"Saved: {out_path}")  # 경로 로그 출력
    else:  # 미저장 모드인 경우
        cv2.imshow("Inference", img_cv)  # 창으로 표시
        cv2.waitKey(0)  # 키 입력 대기
        cv2.destroyAllWindows()  # 창 정리


@torch.no_grad()  # 동영상 파일 추론 시 그래프 비활성화
def infer_video(video_path, model, transform, device, save_dir):  # 비디오 파일 추론 함수
    cap = cv2.VideoCapture(video_path)  # 비디오 캡처 열기
    if not cap.isOpened():  # 실패 시 처리
        print(f"Failed to open video: {video_path}")  # 오류 로그
        return  # 조기 종료

    writer = None  # VideoWriter 지연 초기화 핸들
    out_path = None  # 출력 경로 변수

    while True:  # 프레임 루프
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:  # 더 이상 프레임 없음
            break  # 루프 종료

        # PIL 변환 → transform(Resize+Crop) 일원화  # 학습 일관성 유지
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR→RGB 변환
        img_pil = Image.fromarray(rgb)  # NumPy→PIL 변환
        input_tensor = transform(img_pil).unsqueeze(0).to(device)  # 텐서 변환 및 이동
        angle = clamp_angle(model(input_tensor).item())  # 추론 및 클램프

        vis = draw_overlay(frame.copy(), angle)  # 시각화용 복사본에 오버레이

        # 저장기 초기화는 첫 프레임에서 안전하게  # 실제 해상도/FPS로 설정
        if save_dir and writer is None:  # 저장이 요청되고 아직 생성 안 됨
            os.makedirs(save_dir, exist_ok=True)  # 폴더 생성
            out_path = os.path.join(save_dir, "result.mp4")  # 출력 경로 결정
            fps = cap.get(cv2.CAP_PROP_FPS) or 30  # FPS 추정(0이면 30으로 대체)
            h, w = vis.shape[:2]  # 프레임 크기
            writer = try_open_writer(out_path, fps, (w, h))  # VideoWriter 생성

        if writer:  # 저장 활성화 시
            writer.write(vis)  # 프레임 기록

        cv2.imshow("Inference", vis)  # 화면 출력
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'로 종료
            break  # 루프 탈출

    cap.release()  # 캡처 해제
    if writer:  # 저장기 있으면
        writer.release()  # 파일 마무리
        print(f"Saved: {out_path}")  # 저장 경로 출력
    cv2.destroyAllWindows()  # 모든 창 정리


@torch.no_grad()  # 카메라 스트림 추론 시 그래프 비활성화
def infer_camera_index(cam_index, model, transform, device, save_dir):  # 실시간 카메라 추론 함수
    cap = open_capture(cam_index)  # 지연 최소화 옵션으로 캡처 열기
    if not cap or not cap.isOpened():  # 장치 오픈 실패 검사
        print(f"Failed to open camera index {cam_index}")  # 오류 로그
        return  # 조기 종료

    writer = None  # VideoWriter 핸들
    out_path = None  # 출력 경로 변수

    while True:  # 스트림 루프
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:  # 읽기 실패/종료
            break  # 루프 종료

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 색공간 변환
        img_pil = Image.fromarray(rgb)  # PIL 이미지 생성
        input_tensor = transform(img_pil).unsqueeze(0).to(device)  # 전처리 및 텐서화
        angle = clamp_angle(model(input_tensor).item())  # 추론 및 범위 제한

        vis = draw_overlay(frame.copy(), angle)  # 시각화 프레임 생성

        if save_dir and writer is None:  # 저장 설정이 있고 아직 미생성이라면
            os.makedirs(save_dir, exist_ok=True)  # 폴더 보장
            out_path = os.path.join(save_dir, "camera_result.mp4")  # 출력 경로 설정
            # FPS/Size는 실제 프레임 기준으로 결정  # 장치 보고값 사용
            fps = cap.get(cv2.CAP_PROP_FPS) or 30  # FPS 감지 실패 시 30 적용
            h, w = vis.shape[:2]  # 현재 프레임 크기
            writer = try_open_writer(out_path, fps, (w, h))  # VideoWriter 생성

        if writer:  # 저장기 활성화
            writer.write(vis)  # 프레임 기록

        cv2.imshow("Inference", vis)  # 화면 출력
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 입력으로 종료
            break  # 루프 종료

    cap.release()  # 캡처 해제
    if writer:  # 저장기가 있다면
        writer.release()  # 파일 클로즈
        print(f"Saved: {out_path}")  # 저장 경로 로그
    cv2.destroyAllWindows()  # 창 정리


def infer_folder(folder_path, model, transform, device, save_dir):  # 폴더 내 이미지 일괄 추론
    paths = sorted(glob(os.path.join(folder_path, "*.jpg")))  # JPG 목록 수집
    if not paths:  # JPG가 없다면
        paths = sorted(glob(os.path.join(folder_path, "*.png")))  # PNG 목록 대체 수집
    for path in paths:  # 각 파일 경로에 대해
        infer_image(path, model, transform, device, save_dir)  # 이미지 추론 수행


# ===================== Main =====================  # 진입점 섹션
if __name__ == '__main__':  # 스크립트 직접 실행 시에만 동작
    parser = argparse.ArgumentParser()  # 인자 파서 생성
    parser.add_argument('--input', type=str, required=True, help='Path to image/video/folder or camera index (e.g., "0")')  # 입력 경로/인덱스
    parser.add_argument('--model', type=str, default='weights/best_model.pth', help='Path to model file')  # 모델 경로
    parser.add_argument('--no-save', action='store_true', help='Disable saving output results')  # 저장 비활성화 플래그
    args = parser.parse_args()  # 인자 파싱

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA 가능 시 GPU 사용
    transform = get_transform()  # 전처리 파이프라인 생성
    model = load_model(args.model, device)  # 모델 로드
    save_dir = None if args.no_save else get_unique_exp_folder()  # 저장 폴더 결정

    # 입력 라우팅  # 입력 타입별 분기 처리
    if args.input.isdigit():  # 카메라 인덱스인 경우
        cam_index = int(args.input)  # 정수 변환
        infer_camera_index(cam_index, model, transform, device, save_dir)  # 카메라 추론 실행
    elif os.path.isdir(args.input):  # 디렉터리인 경우
        infer_folder(args.input, model, transform, device, save_dir)  # 폴더 일괄 추론
    elif args.input.lower().endswith(('.jpg', '.png')):  # 이미지 파일인 경우
        infer_image(args.input, model, transform, device, save_dir)  # 단일 이미지 추론
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # 비디오 파일인 경우
        infer_video(args.input, model, transform, device, save_dir)  # 동영상 추론
    else:  # 그 외 포맷은 미지원
        print("Unsupported input type. Provide a video (.mp4), image (.jpg/.png), folder path, or camera index.")  # 사용법 안내 출력