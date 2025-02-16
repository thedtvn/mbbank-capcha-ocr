import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from core import chars, OcrModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(img_path: str):
    model = OcrModel()
    model.load_state_dict(torch.load("test/model_small.pt", map_location=device))
    model.eval()
    img = Image.open(img_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((160, 50)),
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    img = Variable(img).to(device)
    pred = model(img)

    pred_labels = pred.argmax(dim=2)
    pred_text = [''.join([chars[c] for c in pred_label]) for pred_label in pred_labels]
    print(pred_text)
