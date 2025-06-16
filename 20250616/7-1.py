import numpy as np
import cv2 as cv
import sys

# Optical Flow 시각화를 위한 함수
def draw_opticalFlow(img, flow, step=16):
    h, w = img.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = flow[y, x].astype(np.int32)  # np.int는 사용 금지, 대신 np.int32 사용
            end_point = (x + dx, y + dy)
            if dx * dx + dy * dy > 1:  # 움직임 크기가 일정 이상일 경우
                cv.line(img, (x, y), end_point, (0, 0, 255), 2)  # 빨간색 선 (큰 움직임)
            else:
                cv.line(img, (x, y), end_point, (255, 0, 0), 1)  # 파란색 선 (작은 움직임)

# 카메라 연결
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    sys.exit('카메라 연결 실패')

prev = None  # 이전 프레임 저장 변수

while True:
    ret, frame = cap.read()
    if not ret:
        sys.exit("프레임 읽기 실패")

    # 현재 프레임을 그레이스케일로 변환
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 첫 프레임 처리
    if prev is None:
        prev = gray
        continue

    # Optical Flow 계산
    flow = cv.calcOpticalFlowFarneback(prev, gray, None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)

    # Optical Flow 시각화
    draw_opticalFlow(frame, flow)

    # 결과 출력
    cv.imshow('optical flow', frame)

    # 현재 프레임을 이전 프레임으로 저장
    prev = gray

    # 'q' 키를 누르면 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
