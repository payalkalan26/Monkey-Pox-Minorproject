# ===============================
# Monkeypox Detection - Improved Model (v3 Fixed)
# ===============================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# 1️⃣ Load and Preprocess Data
# ===============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.3,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("✅ Class indices:", train_gen.class_indices)

test_gen = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print(f"\n✅ Found {train_gen.samples} training images across {train_gen.num_classes} classes.")
print(f"✅ Found {test_gen.samples} testing images across {test_gen.num_classes} classes.\n")

# ===============================
# 2️⃣ Compute Class Weights
# ===============================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("📊 Class Weights:", class_weights)

# ===============================
# 3️⃣ Build Model (MobileNetV3)
# ===============================

base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True   # Unfreeze for fine-tuning

# Freeze lower layers, fine-tune last 80
for layer in base_model.layers[:-80]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ===============================
# 4️⃣ Compile Model
# ===============================

# You can switch to focal loss if dataset is very imbalanced
use_focal_loss = False

if use_focal_loss:
    from tensorflow_addons.losses import SigmoidFocalCrossEntropy
    loss_fn = SigmoidFocalCrossEntropy()
else:
    loss_fn = 'categorical_crossentropy'

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=loss_fn,
    metrics=['accuracy']
)

# ===============================
# 5️⃣ Callbacks for Stability
# ===============================

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint("models/monkeypox_mobilenetv3_best.h5",
                    save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
]

# ===============================
# 6️⃣ Train the Model
# ===============================

print("🧠 Fine-tuning MobileNetV3...")
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=30,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ===============================
# 7️⃣ Save Final Model
# ===============================

os.makedirs("models", exist_ok=True)
model.save("models/monkeypox_mobilenetv3_final.h5")
print("\n✅ Training completed! Model saved successfully at 'models/monkeypox_mobilenetv3_final.h5'")
print("✅ Class indices:", train_gen.class_indices)

# ===============================
# 8️⃣ Plot Accuracy and Loss
# ===============================

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
print("✅ Class indices:", train_gen.class_indices)
plt.show()
