import cv2
cap = cv2.VideoCapture('demo.mp4')
i=0
while True:
    ret, frame = cap.read()
    cv2.imwrite(str(i)+'.jpg',frame)
    i=i+1