import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, ResNet50
from tensorflow.keras.layers import (Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,
                                     Concatenate, Bidirectional, LSTM, Reshape, Flatten,
                                     MultiHeadAttention, Add, LayerNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from collections import Counter

# Paths
save_dir = "/content/drive/MyDrive/save_models"
model_name = "inception_resnet_limited"
model_path = os.path.join(save_dir, model_name + ".keras")

train_dir = "/content/drive/MyDrive/mri_data/train"
val_dir = "/content/drive/MyDrive/mri_data/val"
test_dir = "/content/drive/MyDrive/mri_data/test"

# Parameters
img_size = (224, 224)
batch_size = 32
epochs = 10

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_data = val_test_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
test_data = val_test_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# Input
input_layer = Input(shape=(224, 224, 3))

# Base models
base1 = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_layer)
base2 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_layer)

# Freeze most layers (reduce trainable capacity)
for layer in base1.layers:
    layer.trainable = False
for layer in base2.layers:
    layer.trainable = False

# Unfreeze very few layers for light fine-tuning
for layer in base1.layers[-5:]:
    layer.trainable = True
for layer in base2.layers[-5:]:
    layer.trainable = True

# Feature extraction
x1 = GlobalAveragePooling2D()(base1.output)
x2 = GlobalAveragePooling2D()(base2.output)
fused = Concatenate()([x1, x2])

# Dense block with strong regularization
fused = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(fused)
fused = Dropout(0.6)(fused)
fused = BatchNormalization()(fused)

# Reshape for LSTM + Attention
reshaped = Reshape((1, 128))(fused)
lstm = Bidirectional(LSTM(32, return_sequences=True))(reshaped)
attn = MultiHeadAttention(num_heads=2, key_dim=16)(lstm, lstm)
attn = Add()([attn, lstm])
attn = LayerNormalization()(attn)

# Final Dense
flat = Flatten()(attn)
dense = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(flat)
drop = Dropout(0.5)(dense)
output = Dense(1, activation='sigmoid')(drop)

# Model compile
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(model_path, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)
]

# Train
history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks)

# Evaluate
loss, acc = model.evaluate(test_data)
print(f"✅ Test Accuracy: {acc*100:.2f}%, Test Loss: {loss:.4f}")

# Predict
y_true = test_data.classes
y_prob = model.predict(test_data).ravel()
y_pred = (y_prob > 0.5).astype(int)

# Report
print(classification_report(y_true, y_pred, target_names=['Healthy', 'MS']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'MS'], yticklabels=['Healthy', 'MS'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Training Curves
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Test Acc/Loss Bars
plt.figure(figsize=(5, 4))
plt.bar(['Test Accuracy', 'Test Loss'], [acc, loss], color=['green', 'red'])
plt.title('Test Accuracy and Loss')
plt.ylim(0, 1)
for i, v in enumerate([acc, loss]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.show()
