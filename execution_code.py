import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.label = tk.Label(self.window, text="Choose an option:")
        self.label.pack(pady=10)
        
        self.camera_button = tk.Button(self.window, text="Open Camera", width=20, command=self.open_camera)
        self.camera_button.pack(pady=5)
        
        self.upload_button = tk.Button(self.window, text="Upload Image", width=20, command=self.upload_image)
        self.upload_button.pack(pady=5)
        
        self.model = self.load_model()  # Load the animal emotion detection model
        
        self.window.mainloop()

    def load_model(self):
        try:
            model = tf.keras.models.load_model('animal_emotion_detection_model.keras')
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return None

    def open_camera(self):
        self.window.withdraw()  # Hide the main window temporarily
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera.")
            self.window.deiconify()  # Restore the main window
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform inference on the frame
            emotions, frame = self.detect_emotion(frame)
            
            cv2.imshow('Camera', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.window.deiconify()  # Restore the main window

    def upload_image(self):
        self.window.withdraw()  # Hide the main window temporarily
        
        file_path = filedialog.askopenfilename()
        if not file_path:
            self.window.deiconify()  # Restore the main window
            return
        
        try:
            image = cv2.imread(file_path)
            emotions, image = self.detect_emotion(image)
            self.display_image(image)
            messagebox.showinfo("Emotions Detected", f"Emotions: {emotions}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")
        
        self.window.deiconify()  # Restore the main window

    def detect_emotion(self, image):
        try:
            resized_image = cv2.resize(image, (224, 224))  # Resize to match model input size
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            resized_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension
            
            # Preprocess image if necessary (e.g., normalize pixel values)
            # Perform prediction
            predictions = self.model.predict(resized_image)
            
            # Process predictions to get emotions (example code, adjust as per your model's output)
            # For example, assuming the model outputs a softmax activation for emotions:
            emotions = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]
            max_index = np.argmax(predictions)
            predicted_emotion = emotions[max_index]
            
            # Example overlay text on image
            cv2.putText(image, predicted_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            return predicted_emotion, image
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting emotion: {e}")
            return None, image

    def display_image(self, image):
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        
        self.image_label = tk.Label(self.window, image=image)
        self.image_label.image = image
        self.image_label.pack(padx=10, pady=10)
        
        close_button = tk.Button(self.window, text="Close", command=self.close_image)
        close_button.pack(pady=5)

    def close_image(self):
        self.image_label.pack_forget()

if __name__ == "__main__":
    window = tk.Tk()
    app = App(window, "Animal Emotion Detection System")
