import cv2 

camera = cv2.VideoCapture(0) 

while(True): 
    ret, frame = camera.read() 

    cv2.imshow('frame', frame) 
    key= cv2.waitKey(0)
    # if the 's' key is pressed, save the image
    if key == ord('s'):
        cv2.imwrite('hand_gesture.jpg', frame)
        print('Image saved')
    
    
# release the camera and close the window
camera.release()
cv2.destroyAllWindows()