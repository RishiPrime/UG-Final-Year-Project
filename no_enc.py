import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import ast

# Initialize MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Create a directory to store captured photos
output_directory = 'C:/Users/Rishi/Dropbox/PC/Desktop/captured_data'
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
  return np.sqrt(((vector1 - vector2)**2).sum())

if choice == '1':
    global name
    name = input("Enter your name: ")  # User enters their name

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

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

                        # Save the landmarks data to a single CSV file
                        csv_filename = os.path.join(output_directory, f'{name}_facial_landmarks.csv')
                        with open(csv_filename, 'w', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([name, landmarks_data])
                            global facial_landmarks_database
                            facial_landmarks_database = [name, landmarks_data]

                        # Save the photo
                        photo_filename = os.path.join(output_directory, f'{name}_captured_photo.png')
                        cv2.imwrite(photo_filename, frame)
                        print(f'Photo saved as {photo_filename}')

                        photo_captured = True
                        # Release the camera after capturing the photo
                        cap.release()

            # Display the frame with landmarks
            cv2.imshow('Face Landmarks', frame)

            key = cv2.waitKey(3)

            if key == 27:  # Press 'Esc' to exit
                break
            elif key == ord('s') and not photo_captured:  # Press 's' to capture a photo (only once)
                photo_captured = True
    
elif choice == '2':
    # Log in with an existing face
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

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
                    # Extract the user's facial landmarks
                    user_landmarks_data = []
                    for landmark in landmarks.landmark:
                        user_landmarks_data.extend([landmark.x, landmark.y, landmark.z])
                    cap.release()
                    key = cv2.waitKey(3)
                    # # Compare the user's facial landmarks to the database
                    # for name, landmarks_data in facial_landmarks_database.items():
                        # Read the landmark data from the CSV file
                    name = input("Enter your name: ")
                    with open(os.path.join(output_directory, f'{name}_facial_landmarks.csv'), 'r') as csvfile:
                        csv_reader = csv.reader(csvfile)
                        landmarks_data = next(csv_reader)[1]
                        landmark_data = ast.literal_eval(landmarks_data)
                    user_landmarks_data_array = np.array(user_landmarks_data)
                    landmark_data_array = np.array(landmark_data)
                    # Compare the user's facial landmarks to the database
                    # if np.linalg.norm(user_landmarks_data_array - landmark_data_array) < 5:
                    #     # Login successful
                    #     print(f'Login successful! Welcome, {name}.')
                    #     break
                    # else:
                    # # Login failed
                    #     print('Login failed.')
                        
                    euclidean_distance = euclidean_distance(user_landmarks_data_array, landmark_data_array)

                    # If the distance is less than 5, the user's face matches the landmark data.
                    if euclidean_distance < 4:
                        print(f'Login successful! Welcome, {name}.')
                    else:
                        print('Login failed.')


# Close OpenCV windows
cv2.destroyAllWindows()