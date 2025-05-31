import argparse
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Simple CNN model for digit recognition

def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(dataset_path, model_path='digit_model.h5'):
    """Train the CNN using images in dataset_path."""
    data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_gen = data_gen.flow_from_directory(
        dataset_path,
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='sparse')

    model = build_model()
    model.fit(train_gen, epochs=5)
    model.save(model_path)
    print(f"model saved to {model_path}")


def predict_digits(image_path, model_path='digit_model.h5'):
    """Read digits from image_path using a trained model."""
    model = keras.models.load_model(model_path)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    for cnt in sorted(contours, key=cv2.boundingRect):
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 10 or w < 5:
            continue
        digit = th[y:y+h, x:x+w]
        digit = cv2.resize(digit, (28, 28))
        digit = digit.astype('float32') / 255.0
        digit = np.expand_dims(digit, axis=-1)
        digits.append((x, digit))

    result = ''
    for x, d in sorted(digits, key=lambda t: t[0]):
        pred = model.predict(d[np.newaxis, ...])
        result += str(np.argmax(pred))
    print(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='path to digit dataset')
    parser.add_argument('--predict', help='path to meter image')
    parser.add_argument('--model', default='digit_model.h5', help='model file')
    args = parser.parse_args()

    if args.train:
        train_model(args.train, args.model)
    if args.predict:
        predict_digits(args.predict, args.model)


if __name__ == '__main__':
    main()
