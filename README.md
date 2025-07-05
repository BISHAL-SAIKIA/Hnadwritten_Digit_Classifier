# 🧠 Handwritten Digit Recognizer with TensorFlow

This project is a simple handwritten digit classifier built using the **MNIST dataset** and **TensorFlow**. It includes:

- `training.py` — to train the model on MNIST
- `main.py` — to test the model with your own digit image (e.g., from phone, scanner, etc.)

---

## 🚀 Demo

![MNIST Example](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

---

## 📁 Project Structure
.
- Digit_Classifier.keras : Saved trained model
- training.py : Model training script
- main.py : Run model on your own images


---

## 🧠 How It Works

- The model is trained using the MNIST dataset (`28x28` grayscale digits).
- The trained model is saved as `Digit_Classifier.keras`.
- `main.py` lets you classify any image of a digit by:
  - Preprocessing the image to MNIST format
  - Running the model prediction
  - Printing the digit and confidence score

---

## 📦 Requirements

Install the following Python packages:

```bash
pip install tensorflow numpy pillow
