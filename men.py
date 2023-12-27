import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import ast
import json
import random
import hashlib
import re


def generate_encryption_key():
    """Generates a random encryption key."""
    return random.SystemRandom().getrandbits(256)


def encrypt_landmark_data(landmark_data, encryption_key):
    """Encrypts the landmark data using the encryption key."""
    encryption_key = str(encryption_key)
    cipher = hashlib.sha256(encryption_key.encode("utf-8")).hexdigest()
    encrypted_landmark_data = []
    landmark_data = str(landmark_data)
    for landmark in landmark_data:
        encrypted_landmark_data.append(cipher[:16] + landmark + cipher[-16:])
    return encrypted_landmark_data


def decrypt_landmark_data(encrypted_landmark_data, encryption_key):
    """Decrypts the landmark data using the encryption key."""
    encryption_key = str(encryption_key)
    cipher = hashlib.sha256(encryption_key.encode("utf-8")).hexdigest()
    decrypted_landmark_data = []
    for landmark in encrypted_landmark_data:
        decrypted_landmark_data.append(landmark[16:-16])
    return decrypted_landmark_data


encryption_key = generate_encryption_key()
# Initialize MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Create a directory to store captured photos
output_directory = "captured_photos"
os.makedirs(output_directory, exist_ok=True)

# Initialize a video capture object
cap = cv2.VideoCapture(0)

photo_count = 0  # Variable to count the captured photos

photo_captured = False  # Flag to indicate if a photo has been captured

# Create a simple text-based menu
print("Welcome!")
print("Choose an option:")
print("1. Register Face")
print("2. Log In")
choice = input("Enter the number of your choice (1 or 2): ")


def euclidean_distance(vector1, vector2):
    if vector1.size > vector2.size:
        vector2 = vector2.reshape(vector1.shape)
    else:
        vector1 = vector1.reshape(vector2.shape)
    return np.sqrt(((vector1 - vector2) ** 2).sum())


if choice == "1":
    global name
    name = input("Enter your name: ")  # User enters their name

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=128
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Face Mesh
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks)

                    if not photo_captured:
                        # Extract and store landmarks
                        landmarks_data = []
                        for landmark in landmarks.landmark:
                            landmarks_data.extend([landmark.x, landmark.y, landmark.z])
                        encrypted_landmark_data = encrypt_landmark_data(
                            landmarks_data, encryption_key
                        )
                        # Save the landmarks data to a single CSV file
                        csv_filename = os.path.join(
                            output_directory, f"{name}_facial_landmarks.csv"
                        )
                        # csvfile = open(csv_filename, 'w', newline='')
                        # csv_writer = csv.writer(csvfile)
                        # csv_writer.writerow([name, encrypted_landmark_data])
                        # csvfile.close()
                        with open(csv_filename, "w", newline="") as csvfile:
                            csv_writer = csv.writer(csvfile)
                            for encoded_value in encrypted_landmark_data:
                                csv_writer.writerow([encoded_value])

                        # Save the photo
                        photo_filename = os.path.join(
                            output_directory, f"{name}_captured_photo.png"
                        )
                        cv2.imwrite(photo_filename, frame)
                        print(f"Photo saved as {photo_filename}")

                        photo_captured = True
                        # Release the camera after capturing the photo
                        cap.release()

            # Display the frame with landmarks
            cv2.imshow("Face Landmarks", frame)

            key = cv2.waitKey(3)

            if key == 27:
                break
            elif key == ord("s") and not photo_captured:
                photo_captured = True

elif choice == "2":
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    user_landmarks_data = []
                    for landmark in landmarks.landmark:
                        user_landmarks_data.extend([landmark.x, landmark.y, landmark.z])
                    cap.release()
                    key = cv2.waitKey(3)
                    name = input("Enter your name: ")
                    with open(
                        os.path.join(output_directory, f"{name}_facial_landmarks.csv"),
                        "r",
                    ) as csvfile:
                        csv_reader = csv.reader(csvfile)
                        encrypted_landmark_data = []
                        for row in csv_reader:
                            encrypted_landmarks_dat = row[0]
                            encrypted_landmark_data.append(encrypted_landmarks_dat)
                    landmarks_data = decrypt_landmark_data(
                        encrypted_landmark_data, encryption_key
                    )
                    # landmark_data = ast.literal_eval(landmarks_data)
                    # print(landmarks_data)
                    landmark_data_object = json.dumps(landmarks_data)
                    user_landmarks_data_array = np.array(user_landmarks_data)

                    # landmark_data_array = np.array(landmark_data)

                    # Compare the user's facial landmarks to the database
                    # if np.linalg.norm(user_landmarks_data_array - landmark_data_array) < 5:
                    #     # Login successful
                    #     print(f'Login successful! Welcome, {name}.')
                    #     break
                    # else:
                    # # Login failed
                    #     print('Login failed.')

                    # print(landmark_data_object)
                    # print(landmark_data_object)

                    landmark_data_object = re.sub(r"[^\d.]", "", landmark_data_object)
                    stri = landmark_data_object.split()
                    landmark_data_object = re.findall(r"0\.\d+", stri[0])
                    fl = [float(s) for s in landmark_data_object]
                    fl_array = np.array(fl)
                    euclidean_distance = euclidean_distance(
                        user_landmarks_data_array[:702], fl_array
                    )

                    # If the distance is less than 5, the user's face matches the landmark data.
                    if euclidean_distance < 10:
                        print(f"Login successful! Welcome, {name}.")
                    else:
                        print("Login failed.")


# Close OpenCV windows
cv2.destroyAllWindows()
