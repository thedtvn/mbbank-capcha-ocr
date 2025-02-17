# OCR Model Training and Prediction

This project is designed to train and use an Optical Character Recognition (OCR) model for recognizing characters in CAPTCHA images.

## Project Structure

- `mb_capcha_ocr/`: Contains the core OCR model and prediction logic.
- `train_model/`: Contains the training script for the OCR model.

## Installation and Setup for Training

1. Clone the repository:
    ```sh
    git clone https://github.com/thedtvn/mbbank-capcha-ocr
    cd mbbank-capcha-ocr
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Training the Model

1. Place your training and testing images in the `dataset/` directory. The images should be named in the format `{label}.(png|jpg|jpeg)`.

2. Run the training script:
    ```sh
    python train_model/train.py
    ```

3. The trained model will be saved as `model.pt` in the root directory.

## Using the Model for Prediction

1. Import the `predict` function from the `mb_capcha_ocr` module:
    ```python
    from mb_capcha_ocr.predict import predict
    ```

2. Use the `predict` function to get the predicted text from an image:
    ```python
    from PIL import Image

    img = Image.open("path_to_image.png")
    predicted_text = predict(img)
    print(predicted_text)
    ```

## Files

- `train_model/train.py`: Script to train the OCR model.
- `mb_capcha_ocr/predict.py`: Script to predict text from an image using the trained OCR model.
- `requirements.txt`: List of dependencies required for the project.

## Dependencies

- Python 3.x
- torch
- torchvision
- matplotlib
- Pillow

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.