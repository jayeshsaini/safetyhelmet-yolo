import cv2
from darkflow.net.build import TFNet
import numpy as np
import time


import face_recognition




# Load a sample picture and learn how to recognize it.
jayesh_image = face_recognition.load_image_file("JayeshSaini.jpg")
jayesh_face_encoding = face_recognition.face_encodings(jayesh_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [jayesh_face_encoding]
known_face_names = ["Jayesh Saini"]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []


#defining options to run the model
#here we pass the arguments to run load the model and weights by running through the directory
options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 5875,
    'threshold': 0.45
}  #can play around with the threshold if you increase it then u will see more boxes

#create tfnet object
tfnet = TFNet(options)

#create some random colors for different colored bounding boxes
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

#craete capture object to open webcam or capture device
capture = cv2.VideoCapture(0)

#set frame size we are going to use full 1080 p
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

process_this_frame = True

while True:
    stime = time.time()
    ret, frame = capture.read()
    #predict results
    results = tfnet.return_predict(frame)  #returns a list
    print(results)
    #if the capture device is still recording we are going to add bounding boxes to our frame and do all the predictions by classifying objects
    #if frame has person wearing specs then only detect face with space and no need to recognise face
    if ret and len(results) > 0 :
    # if ret:    
        for color, result in zip(colors, results):  #for each object we get a unique color
            #topleft
            tl = (result['topleft']['x'], result['topleft']['y'])
            #bottom right
            br = (result['bottomright']['x'], result['bottomright']['y'])

            label = result['label'] #get the prdicted label for the particular objects in frame
            confidence = result['confidence']

            #format text for label and confidence to be displayed
            text = '{}: {:.0f}%'.format(label, confidence * 100)

            #add rectangle on the frame of objects
            frame = cv2.rectangle(frame,tl,br,color,5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

            if (confidence * 100) < 15:

                # frame = frame[:, :, ::-1]
                if process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(frame)
                    face_encodings = face_recognition.face_encodings(frame, face_locations)

                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"

                        # # If a match was found in known_face_encodings, just use the first one.
                        # if True in matches:
                        #     first_match_index = matches.index(True)
                        #     name = known_face_names[first_match_index]

                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        face_names.append(name)
                process_this_frame = not process_this_frame
                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name + ' is not wearing Helmet', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))

    #recognise face if person is not wearing specs
    elif ret and (len(results) == 0):
    
        # frame = frame[:, :, ::-1]
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
        process_this_frame = not process_this_frame
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name + ' is not wearing Helmet', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))


    #boiler plate code to stop video capture
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


