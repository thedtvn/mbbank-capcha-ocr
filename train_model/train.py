import multiprocessing
import os
import sys
from typing import Literal

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

# Add the root folder to the sys path
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_folder)

from core import chars, DatasetLoader, OcrModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset path
# dataset fomart is: {label}.(png|jpg|jpeg)
data_set_path = "dataset"

transform = transforms.Compose([
    transforms.Resize((50, 160)),
    transforms.ToTensor()
])

train_dataset = DatasetLoader(data_set_path, transform=transform)
test_dataset = DatasetLoader(data_set_path, is_test=True, transform=transform)
train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dl = DataLoader(test_dataset)

print("train:", len(train_dataset), "test:", len(test_dataset))

model = OcrModel().to(device)
loss_func = torch.nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_values = []

for epoch in range(100):
    for step, data in enumerate(train_dl):
        img, label_oh, label = data
        img = Variable(img).to(device)
        label_oh = Variable(label_oh.float()).to(device)
        pred = model(img)
        loss = loss_func(pred, label_oh)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        print('epoch:', epoch + 1, 'step:', step + 1, 'loss:', loss.item())

torch_input = torch.randn(1, 1, 50, 160).to(device)
torch.onnx.export(model, (torch_input,), f"modal.onnx", input_names=['input'], output_names=['output'])

plt.plot(loss_values)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title(f'Training Loss')
plt.savefig(f'loss.png')

# Test the model

total = 0
correct = 0

model.eval()

for step, (img, label_oh, label) in enumerate(test_dl):
    img = Variable(img).to(device)
    pred = model(img)

    pred_labels = pred.argmax(dim=2)
    pred_text = [''.join([chars[c] for c in pred_label]) for pred_label in pred_labels]
    if label[0] == pred_text[0]:
        correct += 1
    total += 1

print(f'Accuracy:', correct, '/', total, correct / total)
