import cv2
from cvzone.PoseModule import PoseDetector

# Capture video from webcam
cap = cv2.VideoCapture(0)
detector = PoseDetector()

# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# frame_rate = cap.get(cv2.CAP_PROP_FPS)

# fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # Define codec for MP4 format
# out = cv2.VideoWriter('pose_recording.mp4', fourcc, frame_rate, (frame_width, frame_height))  # Adjust dimensions as needed

while True:
    # Read frame from webcam
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.imshow("webcam stuff", img)
    # lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)
    # print(lmList)
    img = detector.findPose(img)
    lmList, _ = detector.findPosition(img)

    if lmList:
        # Iterate through all landmarks
        for id, landmark in enumerate(lmList):
            # Extract x, y coordinates from the landmark
            x, y, _ = landmark
            # Draw a small circle at each landmark point
            #cv2.circle(img, (x, y), 5, (255, 0, 0), cv2.FILLED)
            # Display landmark ID next to the point
            cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        if 1 - lmList[19][1] > 1 - lmList[11][1]:
            cv2.putText(img, "hand above", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 1, cv2.LINE_AA)

        shirt_path = 'shirt.png'
        img_shirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)

        lm11 = lmList[11][1:3] #shoulder
        lm12 = lmList[12][1:3]

        # width, height = img_shirt.size
        # # height_width_ratio = height / width
        # width_of_shirt = int((lm11[0] - lm12[0]) * height_width_ratio)
        # img_shirt = cv2.resize(img_shirt, (width_of_shirt, int(width_of_shirt * height_width_ratio)))
    # out.write(img)
    # Display frame on screen
    cv2.imshow('Webcam Feed', img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
# out.release()
cv2.destroyAllWindows()