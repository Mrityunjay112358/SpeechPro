# Sign Language to Text and Speech Converter

This project is a web application that converts sign language gestures into text and speech. The application uses a convolutional neural network (CNN) model to recognize sign language gestures from images and provides the corresponding text and speech output in English or Hindi.

## Features
- Upload an image of a sign language gesture.
- Convert the gesture to text.
- Convert the gesture to speech in English or Hindi.

## Technologies Used
- Python
- Flask
- TensorFlow/Keras
- OpenCV
- gTTS (Google Text-to-Speech)
- HTML, CSS, JavaScript

## Setup Instructions

### Prerequisites
- Python 3.x
- Flask
- TensorFlow/Keras
- OpenCV
- gTTS
- mpg321 (for playing audio)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/<repository-name>.git
    cd <repository-name>
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Generate the `X_train.npy` and `y_train.npy` files:
    - Run the `prepare_data.py` script to generate the training data files.
    - Ensure you have the necessary data to generate these files.

    ```bash
    python prepare_data.py
    ```

4. Start the Flask server:
    ```bash
    python app.py
    ```

5. Open `index.html` in your browser.

### Usage
1. Upload an image of a sign language gesture.
2. Select the desired language (English or Hindi).
3. Click "Convert" to see the text and hear the speech output.

## Project Structure
- `app.py`: Flask server and main application logic.
- `index.html`: Front-end HTML file.
- `styles.css`: CSS styles for the front-end.
- `prepare_data.py`: Script to generate training data.
- `train_model.py`: Script to train the model.
- `requirements.txt`: List of Python dependencies.

## Acknowledgements
- [TensorFlow](https://www.tensorflow.org/)
- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)
- [gTTS](https://gtts.readthedocs.io/)

## License
This project is licensed under the MIT License.
