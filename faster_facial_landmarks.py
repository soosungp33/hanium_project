# USAGE
# python faster_facial_landmarks.py --shape-predictor shape_predictor_5_face_landmarks.dat

from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

# dlib의 얼굴 탐지기(HOG 기반)를 초기화 및 얼굴 랜드마크 예측 변수 생성
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Video Stream 초기화
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
#vs = VideoStream(src=1).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

# video stream 반복
while True:
	# 비디오 스트림에서 프레임을 잡고, 최대 너비가 400픽셀이 되도록 크기 조정 후 Grayscale 변환
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# Grayscale 프레임에서 얼굴 감지
	rects = detector(gray, 0)

	# 얼굴 감지되면, 감지된 총 얼굴 수 쓰기
	if len(rects) > 0:
		text = "{} face(s) found".format(len(rects))
		cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# 얼굴 탐지 반복
	for rect in rects:
		# 얼굴의 경계 상자를 계산하여 프레임에 그린다.
		(bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

		# 얼굴 영역의 얼굴 랜드 마크를 결정한 다음 얼굴 랜드 마크 (x, y) 좌표를 NumPy 배열로 변환
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
		# 얼굴 랜드 마크에 대한 (x, y) 좌표를 반복하고 각각을 그린다.
		for (i, (x, y)) in enumerate(shape):
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
			cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# 프레임 Show
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# 'q' key 를 누르면 루프 탈출
	if key == ord("q"):
		break

# cleanup
cv2.destroyAllWindows()
vs.stop()