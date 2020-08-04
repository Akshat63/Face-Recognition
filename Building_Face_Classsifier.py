# Recognise faces using some classification algorithm

# 1- read a video stream and extract faces
# 2 - load the training data (numpy arrays of all the persons)
	#x value is the numpy array and y we need to assign
# 3 - use knn to make prediction	
# 4 - map the prediction id to the name of the user
# 5 - Display predictions on the screen - bounding box and name

import numpy as np 
import cv2
import os

### KNN CODE
def distance(x1,x2):
	return np.sqrt(((x1-x2)**2).sum())

def knn(train,test,k=5):
	dist=[]

	for i in range(train.shape[0]):

		#getting vector and label
		ix=train[i,:-1]
		iy=train[i,-1]

		#computing dist
		d = distance(test,ix)
		dist.append([d,iy])
	#sort based on dist and take top k
	dk = sorted(dist,key=lambda x:x[0])[:k]
	#retrieve only labels

	labels = np.array(dk)[:,-1]
	#getting freq of each label
	output = np.unique(labels,return_counts=True)
	index= np.argmax(output[1])
	return output[0][index]


#init camera
cap = cv2.VideoCapture(0)

# face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
skip = 0

dataset_path='./Data/'

face_data=[]
labels=[]

class_id = 0 # labels for given file
names = {}# mapping  btw id and name

# data preperation 

for fx in os.listdir(dataset_path):
	if fx.endswith('.npy')
		#creates mapping
		names[class_id] = fx[:-4]
		print("Loaded"+fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		#creating labels
		target = class_id*np.ones((data_item.shape[0],))
		class_id+=1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis =1)
print(trainset.shape)

#testing 
while True:

	ret,frame=cap.read()

	if ret==False:
		continue
	faces = face_cascade.detectMultiScale(frame,1.3,5)

	for face in faces:
		x,y,w,h = face

		offset=10
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.reshape(face_section,(100,100))



		#predicted label
		out  = knn(trainset,face_section.flatten())

		# disply name and a rectangle with it
		pred_name = names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)# 2 is the thickness

	cv2.imshow("faces",frame)


	key = cv2.waitKey(1)&0xFF
	if key ==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

