# OCR Model Training and Prediction

This project is designed to train and use an Optical Character Recognition (OCR) model for recognizing characters in
CAPTCHA images.

## Project Structure

- `mb_capcha_ocr/`: Contains the core OCR model and prediction logic.
- `train_model/`: Contains the training script for the OCR model.

## Installation and Setup for Training

1. Clone the repository:
    ```sh
    git clone https://github.com/thedtvn/mbbank-capcha-ocr
    cd mbbank-capcha-ocr
    cd train_model
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r train_requirements.txt
    ```

## Training the Model

1. Place your training and testing images in the `dataset/` directory. The images should be named in the format
   `{label}.(png|jpg|jpeg)`.

2. Run the training script:
    ```sh
    python train.py
    ```

3. The trained model will be saved as `model.onnx` in the directory.

## Using the Model for Prediction

```python
from PIL import Image
from mb_capcha_ocr import OcrModel

model = OcrModel()  # model_path optional if using custom model
img = Image.open("path_to_image.png")
predicted_text = model.predict(img)
print(predicted_text)
```

## Files

- `train_model/train.py`: Script to train the OCR model.
- `mb_capcha_ocr/predict.py`: Script to predict text from an image using the trained OCR model.
- `requirements.txt`: List of dependencies required for the project.

## Dependencies

- Python 3.x
- numpy
- onnxruntime
- Pillow

## Dependencies Training

- Python 3.x
- torch
- torchvision
- matplotlib
- Pillow
- onnx

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Credits

Best thanks to [CookieGMVN](https://github.com/cookieGMVN) for providing
the dataset [V1](https://www.kaggle.com/datasets/cookiegmvn/mbbank-captcha-images) [V2](https://www.kaggle.com/datasets/cookiegmvn/mbbank-captcha-v2).
