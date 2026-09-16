import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --------------------------
# Paths
# --------------------------
model_path = "/content/drive/MyDrive/save_models/inception_resnet_limited.keras"
test_folder = "/content/drive/MyDrive/finaltest"   # <-- your folder

# --------------------------
# Load Model
# --------------------------
model = load_model(model_path)
print("Model loaded successfully!")

# --------------------------
# Parameters
# --------------------------
img_size = (224, 224)
class_names = ["Healthy", "MS"]   # 0 = Healthy, 1 = MS

# --------------------------
# Prediction Function
# --------------------------
def predict_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    pred_class = int(prob > 0.5)

    return pred_class, prob, img


# --------------------------
# Run prediction on all images
# --------------------------
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_name in image_files:
    img_path = os.path.join(test_folder, img_name)

    pred_class, prob, img = predict_image(img_path)

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Image: {img_name}\nPrediction: {class_names[pred_class]}\nConfidence: {prob:.3f}"
    )
    plt.show()
