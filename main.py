import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet121, ResNet50, VGG16, InceptionV3, MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Dataset Loading Function

def load_dataset(csv_path, img_folder):
    df = pd.read_csv(csv_path)


    df = df[
        (df['Label'] == 'Normal') |
        ((df['Label'] == 'Pnemonia') & (df['Label_2_Virus_category'] == 'COVID-19'))
    ]

    # Map labels: Normal → 0, COVID → 1
    df['target'] = df.apply(
        lambda row: 0 if row['Label'] == 'Normal' else 1,
        axis=1
    )

    images, labels = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(img_folder, row['X_ray_image_name'])
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(row['target'])

    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="int32")

    return train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)


# Model Builder Functions

def build_densenet121():
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=x)
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_resnet50():
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=x)
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_vgg16():
    base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=x)
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_inceptionv3():
    base = InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=x)
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_mobilenetv2():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=x)
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Training / Validation

def train_or_load(model_fn, model_filename, X_train, y_train, X_test, y_test, models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_filename)

    if os.path.exists(model_path):
        print(f" Loading existing model: {model_filename}")
        return load_model(model_path)

    print(f" Training new model: {model_filename}")
    model = model_fn()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5, 
        batch_size=32
    )
    model.save(model_path)

    #  Plot Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title(f"{model_filename} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    #  Plot Loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(f"{model_filename} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return model


# Main Function

def main():
    csv_file = "Chest_xray_Corona_Metadata.csv"   
    img_folder = "Coronahack-Chest-XRay-Dataset" 

    X_train, X_test, y_train, y_test = load_dataset(csv_file, img_folder)

    # Run models 
    train_or_load(build_densenet121, "DenseNet121_model.h5", X_train, y_train, X_test, y_test)
    train_or_load(build_resnet50, "ResNet50_model.h5", X_train, y_train, X_test, y_test)
    train_or_load(build_vgg16, "VGG16_model.h5", X_train, y_train, X_test, y_test)
    train_or_load(build_inceptionv3, "InceptionV3_model.h5", X_train, y_train, X_test, y_test)
    train_or_load(build_mobilenetv2, "MobileNetV2_model.h5", X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
