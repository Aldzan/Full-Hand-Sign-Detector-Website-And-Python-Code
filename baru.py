import cv2
import numpy as np
import mediapipe as mp
import os
import pickle
import time


class HandSignDetector:
    def __init__(self, model_path="hand_signs_model.pkl"):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize hands model with max_num_hands=1
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Membatasi deteksi hanya untuk 1 tangan
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Data storage
        self.model_path = model_path
        self.signs_data = {}

        # Load existing model if available
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.signs_data = pickle.load(f)
            print(f"Loaded {len(self.signs_data)} signs from existing model")
        else:
            print("No existing model found. Starting fresh.")

    def save_model(self):
        """Save the current signs data to a file"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.signs_data, f)
        print(f"Model saved to {self.model_path}")

    def extract_hand_landmarks(self, frame):
        """Extract hand landmarks from a frame and calculate bounding boxes"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        landmarks_list = []
        bounding_boxes = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates
                landmarks = []
                x_coordinates = []
                y_coordinates = []

                for lm in hand_landmarks.landmark:
                    x, y, z = lm.x, lm.y, lm.z
                    landmarks.append([x, y, z])

                    h, w, c = frame.shape
                    px, py = int(x * w), int(y * h)
                    x_coordinates.append(px)
                    y_coordinates.append(py)

                if x_coordinates and y_coordinates:
                    min_x, max_x = min(x_coordinates), max(x_coordinates)
                    min_y, max_y = min(y_coordinates), max(y_coordinates)

                    padding = 20
                    min_x = max(0, min_x - padding)
                    min_y = max(0, min_y - padding)
                    max_x = min(w, max_x + padding)
                    max_y = min(h, max_y + padding)

                    bounding_boxes.append((min_x, min_y, max_x, max_y))

                # Normalize coordinates relative to the wrist
                wrist = landmarks[0]  # Wrist is the first landmark
                normalized_landmarks = []
                for lm in landmarks:
                    normalized_landmarks.append([
                        lm[0] - wrist[0],
                        lm[1] - wrist[1],
                        lm[2] - wrist[2]
                    ])

                landmarks_list.append(normalized_landmarks)

        return frame, landmarks_list, bounding_boxes

    def add_new_sign(self, sign_name, num_samples=100):
        """Add a new sign to the model by collecting samples"""
        print(f"Adding new sign: {sign_name}")
        print(f"Prepare to show your hand sign. Collecting {num_samples} samples.")

        cap = cv2.VideoCapture(0)
        samples = []
        sample_count = 0

        # Wait for user to get ready
        print("Get ready...")
        time.sleep(2)
        print("Start showing the sign now!")

        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Extract landmarks
            frame, landmarks_list, bounding_boxes = self.extract_hand_landmarks(frame)

            # Draw bounding boxes for visualization during training
            for i, (min_x, min_y, max_x, max_y) in enumerate(bounding_boxes):
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            if landmarks_list:
                # Use the first detected hand
                samples.append(landmarks_list[0])
                sample_count += 1
                print(f"Sample {sample_count}/{num_samples} collected")

                # Add slight delay between samples
                time.sleep(0.2)

            # Display frame
            cv2.putText(frame, f"Collecting: {sample_count}/{num_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Add New Sign", frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if samples:
            self.signs_data[sign_name] = samples
            print(f"Successfully added {sign_name} with {len(samples)} samples")
            self.save_model()
        else:
            print("Failed to collect samples. No hand was detected.")

    def remove_sign(self, sign_name=None):
        """Remove a sign from the model"""
        if not self.signs_data:
            print("No signs available to remove.")
            return

        if sign_name is None:
            # Display all available signs
            print("Available signs:")
            for i, sign in enumerate(self.signs_data.keys(), 1):
                print(f"{i}. {sign}")

            try:
                choice = input("Enter the number of the sign to remove (or 'q' to cancel): ")
                if choice.lower() == 'q':
                    print("Removal cancelled.")
                    return

                choice = int(choice)
                if 1 <= choice <= len(self.signs_data):
                    # Get the sign name by index
                    sign_name = list(self.signs_data.keys())[choice - 1]
                else:
                    print("Invalid selection.")
                    return
            except ValueError:
                print("Invalid input. Please enter a number.")
                return

        # Remove the sign if it exists
        if sign_name in self.signs_data:
            del self.signs_data[sign_name]
            print(f"Sign '{sign_name}' removed successfully.")
            self.save_model()
        else:
            print(f"Sign '{sign_name}' not found.")

    def reduce_samples(self, sign_name=None, target_samples=15):
        """Reduce the number of samples for a sign"""
        if not self.signs_data:
            print("No signs available.")
            return

        if sign_name is None:
            # Display all available signs
            print("Available signs:")
            for i, sign in enumerate(self.signs_data.keys(), 1):
                sign_samples = len(self.signs_data[list(self.signs_data.keys())[i - 1]])
                print(f"{i}. {sign} ({sign_samples} samples)")

            try:
                choice = input("Enter the number of the sign to reduce samples (or 'q' to cancel): ")
                if choice.lower() == 'q':
                    print("Reduction cancelled.")
                    return

                choice = int(choice)
                if 1 <= choice <= len(self.signs_data):
                    # Get the sign name by index
                    sign_name = list(self.signs_data.keys())[choice - 1]
                else:
                    print("Invalid selection.")
                    return
            except ValueError:
                print("Invalid input. Please enter a number.")
                return

        # Reduce samples for the sign if it exists
        if sign_name in self.signs_data:
            current_samples = len(self.signs_data[sign_name])
            if current_samples <= target_samples:
                print(f"Sign '{sign_name}' already has {current_samples} samples, which is <= target {target_samples}.")
                return

            # Keep only the first target_samples
            self.signs_data[sign_name] = self.signs_data[sign_name][:target_samples]
            print(f"Reduced '{sign_name}' from {current_samples} to {target_samples} samples.")
            self.save_model()
        else:
            print(f"Sign '{sign_name}' not found.")

    def manage_signs(self):
        """Manage the signs in the model"""
        while True:
            print("\n===== Sign Management =====")
            print("1. List all signs")
            print("2. Add a new sign")
            print("3. Remove a sign")
            print("4. Reduce samples for a sign")
            print("5. Return to detection")

            choice = input("Enter your choice (1-5): ")

            if choice == '1':
                if not self.signs_data:
                    print("No signs available.")
                else:
                    print("\nAvailable signs:")
                    for i, sign in enumerate(self.signs_data.keys(), 1):
                        samples = len(self.signs_data[list(self.signs_data.keys())[i - 1]])
                        print(f"{i}. {sign} ({samples} samples)")

            elif choice == '2':
                sign_name = input("Enter name for the new sign: ")
                num_samples = input("Enter number of samples to collect (default: 30): ")
                try:
                    num_samples = int(num_samples)
                except ValueError:
                    num_samples = 30

                self.add_new_sign(sign_name, num_samples)

            elif choice == '3':
                self.remove_sign()

            elif choice == '4':
                target = input("Enter target number of samples (default: 15): ")
                try:
                    target = int(target)
                except ValueError:
                    target = 15

                self.reduce_samples(target_samples=target)

            elif choice == '5':
                print("Returning to detection mode...")
                break

            else:
                print("Invalid choice. Please try again.")

    def detect_sign(self, landmarks):
        """Detect which sign the landmarks match"""
        if not self.signs_data or not landmarks:
            return "Unknown"

        best_match = "Unknown"
        min_distance = float('inf')

        for sign_name, sign_samples in self.signs_data.items():
            # Calculate distance to each sample of this sign
            for sample in sign_samples:
                distance = self.calculate_distance(landmarks, sample)
                if distance < min_distance:
                    min_distance = distance
                    best_match = sign_name

        # Threshold for minimum confidence
        if min_distance > 0.5:
            return "Unknown"

        return best_match

    def calculate_distance(self, landmarks1, landmarks2):
        """Calculate Euclidean distance between two sets of landmarks"""
        if len(landmarks1) != len(landmarks2):
            return float('inf')

        total_distance = 0
        for i in range(len(landmarks1)):
            lm1 = landmarks1[i]
            lm2 = landmarks2[i]

            # Calculate Euclidean distance
            distance = np.sqrt(
                (lm1[0] - lm2[0]) ** 2 +
                (lm1[1] - lm2[1]) ** 2 +
                (lm1[2] - lm2[2]) ** 2
            )
            total_distance += distance

        # Normalize by number of landmarks
        return total_distance / len(landmarks1)

    def run_detection(self):
        """Run the main detection loop"""
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Extract landmarks and bounding boxes
            frame, landmarks_list, bounding_boxes = self.extract_hand_landmarks(frame)

            # Detect sign for each hand and draw bounding boxes
            if landmarks_list:
                for i, (landmarks, box) in enumerate(zip(landmarks_list, bounding_boxes)):
                    sign = self.detect_sign(landmarks)

                    # Draw bounding box
                    min_x, min_y, max_x, max_y = box
                    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

                    # Draw sign text above the bounding box
                    text_position = (min_x, min_y - 10)
                    cv2.putText(frame, sign, text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the available signs (in the corner)
            y_pos = 30
            cv2.putText(frame, "Available Signs:", (frame.shape[1] - 200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            y_pos += 20
            for sign in self.signs_data.keys():
                cv2.putText(frame, f"- {sign}", (frame.shape[1] - 190, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                y_pos += 20

            # Display instructions
            cv2.putText(frame, "Press 'a' to add, 'r' to remove, 'm' to manage signs, 'q' to quit",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show the frame
            cv2.imshow("ASL Hand Sign Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                cap.release()
                cv2.destroyAllWindows()

                sign_name = input("Enter name for the new sign: ")
                self.add_new_sign(sign_name)

                # Restart the camera
                cap = cv2.VideoCapture(0)
            elif key == ord('r'):
                cap.release()
                cv2.destroyAllWindows()

                self.remove_sign()

                # Restart the camera
                cap = cv2.VideoCapture(0)
            elif key == ord('m'):
                cap.release()
                cv2.destroyAllWindows()

                self.manage_signs()

                # Restart the camera
                cap = cv2.VideoCapture(0)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("===== ASL Hand Sign Detection =====")
    print("This program allows you to detect and add custom hand signs.")
    print("Instructions:")
    print("- Press 'a' to add a new sign")
    print("- Press 'r' to remove a sign")
    print("- Press 'm' to manage signs")
    print("- Press 'q' to quit the program")
    print("===================================")

    detector = HandSignDetector()
    detector.run_detection()