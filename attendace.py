import cv2
import face_recognition
import os
from datetime import datetime, timedelta

# Dictionary to store the last attendance timestamp for each person
last_attendance_time = {}

# Function to get encoding of images in a directory
def get_encodings(images_folder):
    known_encodings = []
    known_names = []

    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            img = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(img)

            if face_encodings:
                # Only append if at least one face is detected
                encoding = face_encodings[0]
                known_encodings.append(encoding)
                known_names.append(os.path.splitext(filename)[0])

    return known_encodings, known_names

# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    if name in last_attendance_time:
        last_time = last_attendance_time[name]
        if now - last_time >= timedelta(minutes=5):
            with open("attendance.csv", "a") as file:
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"{name},{timestamp}\n")
                last_attendance_time[name] = now
    else:
        with open("attendance.csv", "a") as file:
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{name},{timestamp}\n")
            last_attendance_time[name] = now

# Main function
def main():
    images_folder = "images"  
    known_encodings, known_names = get_encodings(images_folder)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations,model="cnn")

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            if name != "Unknown":
                mark_attendance(name)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
