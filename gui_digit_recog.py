import tkinter as tk
from tkinter import *

import numpy as np
import cv2
from keras.models import load_model
from keras.optimizers import Adam
from PIL import Image, ImageOps, ImageGrab

model = load_model('mnist.h5', compile=False)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

def process_image(img):
    """ Process image for digit recognition. """
    # Convert to grayscale and apply Gaussian blur
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 0)

    # Threshold the image
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours and extract the bounding box
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        img = img[y:y+h, x:x+w]

    # Resize image to 28x28 pixels
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert colors to match training data
    img = ImageOps.invert(Image.fromarray(img))

    return img

def predict_digit(img):
    img = process_image(img)
    img.save("debug_last_input.png")  # Save the processed image for debugging

    # Convert image to numpy array and normalize
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array.astype('float32') / 255.0

    # Predict the digit
    res = model.predict(img_array)[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.last_x, self.last_y = None, None

        self.canvas = tk.Canvas(self, width=300, height=300, bg="black", cursor="cross")
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_point)

    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text="Draw..")

    def classify_handwriting(self):
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1))

        digit, acc = predict_digit(img)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def draw_lines(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill='white', width=10)
        self.last_x = event.x
        self.last_y = event.y

    def reset_last_point(self, event):
        self.last_x, self.last_y = None, None

app = App()
mainloop()
