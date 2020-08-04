# Read a video stream from camera frame by frame 

import cv2 
cap = cv2.VideoCapture(0) # 0 is the id of web cam if there is only one web cam

while True:

	ret,frame = cap.read()
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if ret==False:
		continue

	cv2.imshow('Video Frame',frame)
	cv2.imshow('Gray Frame',gray_frame)

	# Loop will run till user inputs q
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()
