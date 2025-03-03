import os
import string

import numpy as np
import onnxruntime
from PIL import Image
from torchvision import transforms

dir_path = os.path.dirname(os.path.realpath(__file__))

class OcrModel:

    chars = list(string.digits + string.ascii_letters)

    def __init__(self, model_path: str=None):
        print(self.chars)
        self.session = onnxruntime.InferenceSession(model_path or f"{dir_path}/model.onnx")

    def predict(self, img_data: Image):
        img = img_data.convert("L")
        img = img.resize((160, 50))
        img.show()
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)

        input_name = self.session.get_inputs()[0].name
        pred = self.session.run(None, {input_name: img})[0]

        pred_labels = np.argmax(pred, axis=2)
        pred_text = [''.join([self.chars[c] for c in pred_label]) for pred_label in pred_labels]
        return pred_text

if __name__ == "__main__":
    model = OcrModel()
    img = Image.open("download.png")
    print(model.predict(img))

