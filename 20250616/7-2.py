import numpy as np
import cv2 as cv

# 비디오 파일 열기
cap = cv.VideoCapture('slow_traffic_small.mp4')

# 코너 검출 파라미터 설정
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Lucas-Kanade Optical Flow 파라미터
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 추적 포인트 색상
color = np.random.randint(0, 255, (100, 3))

# 첫 프레임 읽기 및 초기화
ret, old_frame = cap.read()
if not ret:
    print("비디오를 읽을 수 없습니다.")
    cap.release()
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 결과를 그릴 mask 이미지
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    new_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Optical Flow 계산
    p1, status, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    if p1 is not None and status is not None:
        good_new = p1[status == 1]
        good_old = p0[status == 1]

        for i in range(len(good_new)):
            a, b = int(good_new[i][0]), int(good_new[i][1])
            c, d = int(good_old[i][0]), int(good_old[i][1])
            mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)

        img = cv.add(frame, mask)
        cv.imshow('LK tracker', img)

        # 업데이트
        old_gray = new_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # 'q' 키 누르면 종료
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
