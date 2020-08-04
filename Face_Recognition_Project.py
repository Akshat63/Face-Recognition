#Code will capture images from your webcam video stream
# Will extract all faces from the image frame(haarcascades)
# Storing the image in the form on numpy aray

# 1. Capture img, read and show video stream
# 2. Detect faces and show the boundary box
# 3. Flatten the large image(grayscale) and save it in numpy arr form
# 4. Repeating the above process for multiple ppl to form training data


import cv2
import numpy as np 

#init camera
cap = cv2.VideoCapture(0)

# face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
skip = 0
face_data=[]
dataset_path='./Data/'

file_name=input("Enter name of person :  ")
while True:

	ret,frame = cap.read()
	if ret==False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	
	faces = face_cascade.detectMultiScale(frame,1.3,5) # scaling parameter and num of neighbours
	faces = sorted(faces,key=lambda f:f[2]*f[3])
	
	face_section=frame
	# pic last face as it is largest
	for fac in faces[-1:]:
		x,y,w,h=fac # we will sort on the basis of h w
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#extract the required part
		offset = 10 # pixels

		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))
		skip+=1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))

	
	cv2.imshow("Frame",gray_frame)
	cv2.imshow("Face Section",face_section)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed== ord('q'):
		break
# convert face data to np array

face_data= np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))

print(face_data.shape)
np.save(dataset_path+file_name+'.npy',face_data)
print("Data saved successfully")

cap.release()
cv2.destroyAllWindows()