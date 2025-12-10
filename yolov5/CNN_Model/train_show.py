import os  # 운영체제 경로 및 파일 처리 모듈
import pandas as pd  # 데이터프레임 처리를 위한 pandas
import numpy as np  # 수치 계산용 numpy
from PIL import Image  # 이미지 처리를 위한 Pillow
import torch  # PyTorch 메인 라이브러리
import torch.nn as nn  # 신경망 레이어와 손실함수 모듈
from torch.utils.data import Dataset, DataLoader, random_split  # 데이터셋 및 분할/로더 유틸리티
import torchvision.transforms as transforms  # 데이터 전처리 변환 모듈
import matplotlib.pyplot as plt  # 시각화를 위한 matplotlib
from tqdm import tqdm  # 진행률 표시 바
import cv2  # OpenCV: 영상 처리 및 시각화
from model import SteeringModel, get_unique_train_folder  # 사용자 정의 모델과 학습 폴더 생성기
from utils import SteeringDataset  # 사용자 정의 데이터셋 클래스
from config import *  # 설정 상수 불러오기

# 학습 함수 정의
def train(model, loader, optimizer, criterion, device, epoch):
    model.train()  # 모델을 학습 모드로 전환
    running_loss = 0.0  # 손실 누적 변수 초기화

    for i, (imgs, angles) in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)):  # 배치 단위 학습 루프
        imgs, angles = imgs.to(device), angles.to(device)  # 데이터를 디바이스로 이동
        optimizer.zero_grad()  # 이전 기울기 초기화
        preds = model(imgs).squeeze()  # 모델 예측 수행
        loss = criterion(preds, angles)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트
        running_loss += loss.item() * imgs.size(0)  # 배치 손실 누적

        # === 실시간 시각화 (저장 없이 화면 출력만) ===
        img_vis = imgs[0].detach().cpu().permute(1, 2, 0).numpy()  # 첫 번째 이미지를 numpy로 변환
        img_vis = (img_vis * 0.5 + 0.5) * 255  # 정규화 해제 후 0~255 범위로 복원
        img_vis = img_vis.astype(np.uint8)  # 정수형 변환
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)  # RGB → BGR 변환

        h = img_vis.shape[0]  # 이미지 높이
        img_vis[0:120, :] = cv2.GaussianBlur(img_vis[0:120, :], (15, 15), 0)  # 상단 영역 블러 처리
        cv2.rectangle(img_vis, (0, h - 100), (img_vis.shape[1], h), (0, 0, 255), 2)  # 하단 강조 박스

        pred_angle = preds[0].item()  # 첫 번째 샘플의 예측 각도
        cv2.putText(img_vis, f"Pred: {pred_angle:.2f}", (10, 20),  # 예측 각도 표시
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Training Preview", img_vis)  # 실시간 학습 미리보기 창 출력
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 입력 시 종료
            break

    return running_loss / len(loader.dataset)  # 평균 손실 반환

# 하단 crop 함수 정의
def crop_bottom(img):
    img = img.resize((RESIZE_WIDTH, RESIZE_HEIGHT))  # 크기 조정
    return img.crop((0, 120, 320, 180))  # 하단 60픽셀 crop

# 평가 함수 정의
def evaluate(model, loader, criterion, device):
    model.eval()  # 모델을 평가 모드로 전환
    total_loss = 0.0  # 손실 누적값
    preds, labels = [], []  # 예측값과 실제값 저장 리스트
    with torch.no_grad():  # 평가 시 그래프 비활성화
        for imgs, angles in tqdm(loader, desc="Evaluating", leave=False):  # 배치별 평가 루프
            imgs, angles = imgs.to(device), angles.to(device)  # 데이터를 디바이스로 이동
            output = model(imgs).squeeze()  # 모델 예측 수행
            loss = criterion(output, angles)  # 손실 계산
            total_loss += loss.item() * imgs.size(0)  # 손실 누적
            preds.extend(output.cpu().numpy())  # 예측 결과 저장
            labels.extend(angles.cpu().numpy())  # 실제 각도 저장
    return total_loss / len(loader.dataset), preds, labels  # 평균 손실, 예측, 실제 반환

# 메인 실행부
if __name__ == '__main__':
    transform = transforms.Compose([  # 데이터 변환 파이프라인 정의
        transforms.Lambda(crop_bottom),  # 하단 crop 적용
        transforms.ToTensor(),  # 텐서 변환
        transforms.Normalize([0.5]*3, [0.5]*3)  # 정규화
    ])

    dataset = SteeringDataset(LABELS_CSV, DATASET_DIR, transform)  # 데이터셋 로드
    # 학습/테스트 데이터 분할 (8:2)
    train_set, test_set = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    # 데이터 로더 생성
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 학습 디바이스 설정
    model = SteeringModel().to(device)  # 모델 초기화 및 디바이스 할당
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam 옵티마이저
    criterion = nn.MSELoss()  # 손실 함수: 평균제곱오차

    save_dir = get_unique_train_folder()  # 고유 학습 결과 폴더 생성
    log_path = os.path.join(save_dir, 'log.csv')  # 로그 파일 경로 설정
    with open(log_path, 'w') as f:  # 로그 파일 초기화
        f.write('epoch,train_loss,test_loss\n')  # 헤더 작성

    best_loss = float('inf')  # 초기 최적 손실 무한대
    for epoch in range(1, EPOCHS + 1):  # 에폭 반복
        print(f"Epoch {epoch}/{EPOCHS}")  # 현재 에폭 출력
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch)  # 학습 수행
        test_loss, preds, labels = evaluate(model, test_loader, criterion, device)  # 평가 수행
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")  # 손실 출력
        with open(log_path, 'a') as f:  # 로그 파일에 기록
            f.write(f"{epoch},{train_loss:.6f},{test_loss:.6f}\n")

        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth")  # 체크포인트 경로
        torch.save(model.state_dict(), ckpt_path)  # 현재 모델 저장
        if epoch > 1:  # 이전 체크포인트 삭제
            prev_ckpt = os.path.join(save_dir, f"checkpoint_epoch{epoch - 1}.pth")
            if os.path.exists(prev_ckpt):
                os.remove(prev_ckpt)

        if test_loss < best_loss:  # 최적 모델 갱신 조건
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))  # 최적 모델 저장

        torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))  # 마지막 모델 저장

    
    # 손실 곡선 플로팅
    log_df = pd.read_csv(log_path, index_col=False)   # CSV 읽기 (epoch이 인덱스로 잡히지 않도록 설정)
    log_df.columns = log_df.columns.str.strip()      # 열 이름의 앞뒤 공백 제거
    log_df = log_df.astype(float)                    # 모든 열을 실수형(float)으로 변환
    
    plt.figure(figsize=(10,5))                       # 플롯 크기 설정 (10x5 인치)
    
    plt.plot(log_df['epoch'].to_numpy(),             # x축: epoch
             log_df['train_loss'].to_numpy(),        # y축: 학습 손실
             label='Training Loss')                  # 라벨: Training Loss
    
    plt.plot(log_df['epoch'].to_numpy(),             # x축: epoch
             log_df['test_loss'].to_numpy(),         # y축: 검증 손실
             label='Validation Loss', linestyle='--')# 라벨: Validation Loss (점선)
    
    plt.xlabel('Epoch')                              # x축 레이블
    plt.ylabel('Loss')                               # y축 레이블
    plt.title('Training Loss vs Validation Loss')    # 그래프 제목
    plt.legend()                                     # 범례 표시
    plt.grid()                                       # 그리드 표시
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))  # 결과 이미지를 파일로 저장
    plt.close()                                      # 플롯 닫기

    cv2.destroyAllWindows()  # OpenCV 창 정리