import sys
import typing

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from core import chars, DatasetLoader, OcrModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_set_path = "data"

def train(size: typing.Literal["small", "medium", "large"]):
    transform = transforms.Compose([
        transforms.Resize((160, 50)),
        transforms.ToTensor()
    ])

    train_dataset = DatasetLoader(data_set_path, size, transform=transform)
    test_dataset = DatasetLoader(data_set_path, size, is_test=True, transform=transform)

    print("train:", len(train_dataset), "test:", len(test_dataset))

    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_dataset)

    model = OcrModel().to(device)
    loss_func = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_values = []

    for epoch in range(50):
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
            print('eopch:', epoch + 1, 'step:', step + 1, 'loss:', loss.item())

    torch.save(model.state_dict(), f"model_{size}.pt")

    plt.plot(loss_values)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({size})')
    plt.savefig(f'loss_{size}.png')

    # Test

    total = 0
    correct = 0

    model.eval()

    for step, (img, label_oh, label) in enumerate(test_dl):
        img = Variable(img).to(device)
        pred = model(img)

        pred_labels = pred.argmax(dim=2)
        pred_text = [''.join([chars[c] for c in pred_label]) for pred_label in pred_labels]
        print('Correct:', label, pred_text[0])
        if label == pred_text[0]:

            correct += 1
        total += 1

    print(f'Accuracy {size}:', correct, '/', total, correct / total)

if __name__ == '__main__':
    size = sys.argv[1]
    print("Start train", size, "model")
    train(size)
