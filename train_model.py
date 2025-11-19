# train_model.py
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Optional: try sklearn compute_class_weight (works if sklearn installed)
try:
    from sklearn.utils.class_weight import compute_class_weight as sk_compute_class_weight
except Exception:
    sk_compute_class_weight = None

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4

# IMPORTANT: use the actual train/test directories you have on disk
TRAIN_DIR = os.path.join("data", "Train")   # e.g. data\Train\Monkeypox and data\Train\Others
TEST_DIR  = os.path.join("data", "Test")    # e.g. data\Test\Monkeypox and data\Test\Others

MODEL_SAVE_PATH = os.path.join("models", "monkeypox_mobilenetv3_final.h5")
SEED = 42

def compute_class_weights_from_generator(generator):
    """
    Compute class weights from a keras generator (fallback if sklearn not available).
    generator.classes is an array of class indices for each sample.
    """
    class_counts = np.bincount(generator.classes)
    n_classes = len(class_counts)
    total = len(generator.classes)
    # classical heuristic: weight_i = total / (n_classes * count_i)
    class_weights = {i: total / (n_classes * count) if count > 0 else 0.0
                     for i, count in enumerate(class_counts)}
    return class_weights

def safe_compute_class_weights(generator):
    """Try sklearn's compute_class_weight first, otherwise fallback."""
    labels = generator.classes
    classes = np.unique(labels)
    if sk_compute_class_weight is not None:
        try:
            weights = sk_compute_class_weight('balanced', classes=classes, y=labels)
            return {int(c): float(w) for c, w in zip(classes, weights)}
        except Exception:
            pass
    # fallback
    return compute_class_weights_from_generator(generator)

def create_model():
    # ------------ Data generators (train / test) ------------
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=25,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Quick sanity checks: directories exist?
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    print("Loading training data from:", TRAIN_DIR)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',   # two classes
        shuffle=True,
        seed=SEED
    )

    print("Loading test/validation data from:", TEST_DIR)
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Print dataset summary and class indices -> critically important for web app mapping
    print("✅ Train samples:", train_generator.samples)
    print("✅ Train classes (per-folder mapping):", train_generator.class_indices)
    print("✅ Validation samples:", validation_generator.samples)
    print("✅ Validation classes (per-folder mapping):", validation_generator.class_indices)
    # NOTE: train_generator.class_indices and validation_generator.class_indices should match.

    # --------------- Class weights (to handle imbalance) ---------------
    class_weights = safe_compute_class_weights(train_generator)
    print(f"✅ Computed class weights: {class_weights}")

    # --------------- Build model (MobileNetV3Small backbone) ---------------
    print("Creating model...")
    base_model = MobileNetV3Small(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False  # freeze backbone first

    x = base_model.output
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)   # binary output (sigmoid)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # callbacks
    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_auc', mode='max', save_best_only=True)
    ]

    # --------------- Train ---------------
    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Ensure models folder exists and save final model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # --------------- Plot training history ---------------
    plt.figure(figsize=(14, 5))

    # Loss subplot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.legend()

    # AUC subplot
    plt.subplot(1, 3, 2)
    if 'auc' in history.history:
        plt.plot(history.history['auc'], label='train auc')
        plt.plot(history.history['val_auc'], label='val auc')
    plt.title('AUC')
    plt.legend()

    # Precision/Recall subplot
    plt.subplot(1, 3, 3)
    if 'precision' in history.history:
        plt.plot(history.history['precision'], label='train precision')
        plt.plot(history.history['val_precision'], label='val precision')
    if 'recall' in history.history:
        plt.plot(history.history['recall'], label='train recall')
        plt.plot(history.history['val_recall'], label='val recall')
    plt.title('Precision & Recall')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == "__main__":
    create_model()
