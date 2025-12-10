import os  # 파일 경로를 다루기 위한 표준 라이브러리
import pandas as pd  # CSV 파일을 다루기 위한 라이브러리
import torch  # PyTorch 프레임워크
from PIL import Image  # 이미지 처리를 위한 라이브러리
from torch.utils.data import Dataset  # PyTorch Dataset 클래스 상속

# 자율주행 조향 데이터셋 클래스 정의
class SteeringDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)  # CSV 파일 읽어서 pandas DataFrame으로 저장
        self.img_dir = img_dir  # 이미지 파일들이 저장된 디렉토리 경로
        self.transform = transform  # 이미지 전처리 함수 (transform pipeline)

    def __len__(self):
        return len(self.data)  # 전체 데이터 샘플 수 반환

    def __getitem__(self, idx):
        row = self.data.iloc[idx]  # 해당 인덱스에 위치한 데이터 한 줄 추출
        image_path = os.path.join(self.img_dir, row['filename'])  # 이미지 파일 경로 구성
        image = Image.open(image_path).convert('RGB')  # 이미지 로드 후 RGB로 변환
        angle = float(row['steering'])  # 조향각을 실수형으로 변환
        if self.transform:
            image = self.transform(image)  # 전처리 적용
        return image, torch.tensor(angle, dtype=torch.float32)  # 이미지와 조향각 반환