import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#AI method processing
def AI_processing():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalize the data (convert values from [0, 255] to [0, 1])
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # 10 classes for CIFAR-10
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Summarize the model
    model.summary()

    # Train the model
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy}")
    print("Processing done")


#main method
def beginners_main_entry():
    #welcome note
    print("AI beginners Course\n")
    #run AI sample
    AI_processing()


if __name__=="__main__":
    beginners_main_entry()