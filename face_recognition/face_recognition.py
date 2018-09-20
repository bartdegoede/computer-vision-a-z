import cv2

# load cascade files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# take both grayscale and original image
def detect(gray, frame):
    # scale b&w images by a factor of 1.3
    # need at least 5 neigbouring zones to also be accepted
    # returns a tuple of (x, y, w, h); top left corner coordinates
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        # draw rectangles for each detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # detect eyes only if we've detected a face (only on RoI of the face)
        roi_gray = gray[y : y + h, x : x + w]
        roi_colour = frame[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        # draw each detected eye within the face
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_colour, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame


# access webcam
video_capture = cv2.VideoCapture(0)

# apply detect() on the frames coming from the webcam
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
