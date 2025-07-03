"""
Model definitions and training pipelines for drowsiness detection
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import load_dataset_cnn

# Use tf.keras.* for all Keras components
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
ResNet50V2 = tf.keras.applications.ResNet50V2
Dense = tf.keras.layers.Dense
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Dropout = tf.keras.layers.Dropout
Model = tf.keras.models.Model
Adam = tf.keras.optimizers.Adam

def train_model_resnet50v2(data_path, epochs=30, save_path='models/drowsiness_resnet50v2.h5'):
    """Trains a ResNet50V2-based model with strong augmentation, class weighting, and two-stage fine-tuning."""
    img_size = (224, 224)
    batch_size = 32
    num_classes = 4
    class_names = ['Closed', 'Open', 'no_yawn', 'yawn']

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))

    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=img_size + (3,))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n🔁 Training top layers...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        class_weight=class_weights,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )

    for layer in base_model.layers[-50:]:
        layer.trainable = True
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n🔁 Fine-tuning last 50 ResNet layers...")
    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)
        ]
    )

    model.save(save_path)
    print(f"Model saved as {save_path}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'] + history_ft.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'] + history_ft.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'] + history_ft.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'] + history_ft.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/training_metrics_optimized.png')
    plt.close()

    val_gen.reset()
    y_pred = model.predict(val_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_gen.classes
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("plots/confusion_matrix_optimized.png")
    plt.close()
    return model, history, history_ft

def train_model_cnn(data_path, epochs=50, save_path='models/drowsiness_cnn.h5'):
    img_size = (150, 150)
    images, labels = load_dataset_cnn(data_path, img_size)
    images = np.expand_dims(images, axis=-1)
    x_train, x_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, shuffle=True, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=4)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)
    def random_contrast(image, lower=0.9, upper=1.1):
        return tf.image.random_contrast(image, lower=lower, upper=upper)
    def custom_preprocess(image):
        image = random_contrast(image)
        return image
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.9, 1.1],
        preprocessing_function=custom_preprocess
    )
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
    val_generator = val_test_datagen.flow(x_val, y_val, batch_size=32)
    test_generator = val_test_datagen.flow(x_test, y_test, batch_size=32)
    input_shape = images.shape[1:]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )
    model.save(save_path)
    print(f"Model saved as {save_path}")
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/training_history_cnn.png')
    plt.close()
    # Evaluate and plot confusion matrix
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Closed', 'Open', 'no_yawn', 'yawn'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix_cnn.png')
    plt.close()
    print(classification_report(y_true_classes, y_pred_classes, target_names=['Closed', 'Open', 'no_yawn', 'yawn']))
    return model, history
