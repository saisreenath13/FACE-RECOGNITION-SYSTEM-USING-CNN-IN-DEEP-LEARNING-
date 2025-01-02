
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import joblib

# Step 1: Data Collection
def collect_faces():
    username = input("Enter your username: ")
    dir_path = f"Faces/{username}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while count < 100:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            cv2.imwrite(f"{dir_path}/{count}.jpg", face)
            count += 1
        cv2.imshow("Capturing Images", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Step 2: Data Preprocessing
def preprocess_data():
    faces, labels = [], []
    dataset_path = "Faces"
    for folder in os.listdir(dataset_path):
        for image_name in os.listdir(f"{dataset_path}/{folder}"):
            img_path = f"{dataset_path}/{folder}/{image_name}"
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(image, (100, 100))
            faces.append(resized_img)
            labels.append(folder)
    faces = np.array(faces) / 255.0  # Normalize
    faces = faces.reshape(-1, 100, 100, 1)  # Reshape for CNN
    return faces, labels

# Step 3: Create CNN Model
def create_model(num_classes):
    if num_classes is None:
        raise ValueError("The number of classes (num_classes) must be defined!")

    # Your model creation code here
    model = None  # Replace this with your model creation logic
    return model

# Define the number of classes
num_classes = 10  # Replace with the actual number of classes

# Create the model
model = create_model(num_classes)


model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Train the Model
def train_model(faces, labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)
    model = create_model(input_shape=(100, 100, 1), num_classes=len(set(labels)))
    
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save("face_recognition_model.h5")
    joblib.dump(label_encoder, "label_encoder.pkl")
    return model, label_encoder

# Step 5: Real-Time Face Recognition
def recognize_face(model, label_encoder):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face, (100, 100))
            reshaped_face = resized_face.reshape(1, 100, 100, 1) / 255.0
            
            predictions = model.predict(reshaped_face)
            label = label_encoder.inverse_transform([np.argmax(predictions)])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Main Execution Example
if __name__ == "__main__":
    print("1. Collect Face Data")
    print("2. Train Model")
    print("3. Recognize Face")
    choice = int(input("Enter your choice: "))
    
    if choice == 1:
        collect_faces()
    elif choice == 2:
        faces, labels = preprocess_data()
        num_classes = len(set(labels))  # Calculate the number of unique classes
        train_model(faces, labels)
    elif choice == 3:
        model = create_model()
        model.load_weights("face_recognition_model.h5")
        label_encoder = joblib.load("label_encoder.pkl")
        recognize_face(model, label_encoder)
    else:
        print("Invalid choice!")
