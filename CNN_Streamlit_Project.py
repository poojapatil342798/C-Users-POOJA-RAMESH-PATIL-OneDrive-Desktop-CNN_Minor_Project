
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import sys


# ==========================================
# PART A - CNN IMAGE CLASSIFICATION
# ==========================================

def run_cnn():

    # STEP 1 - LOAD MNIST DATASET

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("MNIST Dataset Loaded Successfully!")

    # STEP 2 - DISPLAY DATASET INFORMATION

    print("Training images shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing images shape:", x_test.shape)
    print("Testing labels shape:", y_test.shape)

    print("Number of classes:", len(np.unique(y_train)))
    print("Class names:", np.unique(y_train))

    # STEP 3 - DISPLAY SAMPLE IMAGES

    plt.figure(figsize=(10, 5))

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i], cmap="gray")
        plt.title("Label: " + str(y_train[i]))
        plt.axis("off")

    plt.suptitle("Sample MNIST Handwritten Digits")
    plt.tight_layout()
    plt.show()

    # STEP 4 - IMAGE PREPROCESSING

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print("\nImage Preprocessing Completed!")
    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)
    print("Minimum pixel value:", x_train.min())
    print("Maximum pixel value:", x_train.max())

    # STEP 5 - CREATE CNN MODEL

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    print("\nCNN MODEL SUMMARY")
    model.summary()

    # STEP 6 - COMPILE CNN MODEL

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # STEP 7 - TRAIN CNN MODEL

    print("\nStarting CNN Model Training...")

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1
    )

    print("\nCNN Model Training Completed!")

    # STEP 8 - EVALUATE CNN MODEL

    print("\nEvaluating CNN Model...")

    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    print("\nCNN MODEL EVALUATION RESULTS")
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy * 100, "%")

    # STEP 9 - ACTUAL VS PREDICTED RESULTS

    predictions = model.predict(x_test[:10])

    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 5))

    for i in range(10):
        plt.subplot(2, 5, i + 1)

        plt.imshow(x_test[i].reshape(28, 28), cmap="gray")

        plt.title(
            f"Actual: {y_test[i]}\nPredicted: {predicted_labels[i]}"
        )

        plt.axis("off")

    plt.suptitle("Actual vs Predicted MNIST Digits")
    plt.tight_layout()
    plt.show()


# ==========================================
# PART B - STREAMLIT STUDENT MARKS CALCULATOR
# ==========================================

def run_streamlit_app():

    st.title("Student Marks Calculator")

    st.write(
        "Enter your marks in five subjects to calculate your result."
    )

    # STUDENT DETAILS

    st.subheader("Student Details")

    student_name = st.text_input("Enter Student Name:")

    # SUBJECT MARKS

    st.subheader("Enter Subject Marks")

    subject1 = st.number_input(
        "Subject 1:", min_value=0, max_value=100, step=1
    )

    subject2 = st.number_input(
        "Subject 2:", min_value=0, max_value=100, step=1
    )

    subject3 = st.number_input(
        "Subject 3:", min_value=0, max_value=100, step=1
    )

    subject4 = st.number_input(
        "Subject 4:", min_value=0, max_value=100, step=1
    )

    subject5 = st.number_input(
        "Subject 5:", min_value=0, max_value=100, step=1
    )

    # CALCULATE RESULT

    if st.button("Calculate Result"):

        if student_name.strip() == "":
            st.error("Please enter student name.")

        else:

            marks = [
                subject1,
                subject2,
                subject3,
                subject4,
                subject5
            ]

            total = sum(marks)
            percentage = total / 5

            st.subheader("Student Result")

            st.write("Student Name:", student_name)
            st.write("Total Marks:", total, "/ 500")
            st.write("Percentage:", percentage, "%")

            # GRADE CALCULATION

            if percentage >= 90:
                grade = "A+"

            elif percentage >= 80:
                grade = "A"

            elif percentage >= 70:
                grade = "B"

            elif percentage >= 60:
                grade = "C"

            elif percentage >= 40:
                grade = "D"

            else:
                grade = "F"

            st.write("Grade:", grade)

            # PASS OR FAIL

            if all(mark >= 35 for mark in marks):
                st.success("Result: PASS")

            else:
                st.error("Result: FAIL")


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    if "--train-cnn" in sys.argv:
        run_cnn()

    else:
        run_streamlit_app()
