import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch import max

from .custom_dataset import CustomDataset
from .models import MyResNet


MODEL_SAVE_PATH = 'final_model.pt'

label_encoding = {
    'Canopy Closed': 0,
    'Not Planted': 1,
    'Emerged': 2,
    'Planted': 3,
    'Harvested': 4,
    'Drydown': 5
}


def train(epochs: int, batch_size: int = 32, print_every: int = 1) -> None:
    """
    Implements training of ResNet-18 model for the field state classification problem.
    After training, the best model's weights are saved to the disk.

    Parameters
    ----------
    epochs : int
        Number of epochs to train the model.

    batch_size : int, optional
        Batch size.

    print_every : int, optional
        An integer number indicating how often to print intermediate results (e.g. loss) during each epoch.
        An argument equal to k means print the results after processing k batches.
    """
    train_dataset = CustomDataset('train.csv', label_encoding)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

    test_dataset = CustomDataset('test.csv', label_encoding)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

    net = MyResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    nb_steps = len(train_loader)

    running_loss_train = 0
    best_accuracy_val = 0
    best_model = None

    for e in range(epochs):
        start = time.time()
        net.train()

        n_correct_train = 0
        train_loss = 0
        for step, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            running_loss_train += loss.item()

            net.eval()
            with torch.no_grad():
                _, predicted = max(F.softmax(output, dim=1).data, 1)
                n_correct_train += (predicted == labels).sum().item()

                if (step + 1) % print_every == 0:
                    print(
                        "Epoch: {}/{}".format(e + 1, epochs),
                        "Step: {}/{}".format(step + 1, nb_steps),
                        "Loss: {:.4f}".format(running_loss_train / print_every),
                        "{:.3f} s/{} steps".format((time.time() - start), print_every)
                    )
                    running_loss_train = 0
                    start = time.time()

        with torch.no_grad():
            n_correct_test = 0
            val_loss = 0
            for step, (images, labels) in enumerate(test_loader):
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)

                output = net(images)
                loss = criterion(output, labels)
                val_loss += loss.item()
                _, predicted = max(F.softmax(output, dim=1).data, 1)
                n_correct_test += (predicted == labels).sum().item()

            train_accuracy = 100 * n_correct_train / len(train_dataset)
            val_accuracy = 100 * n_correct_test / len(test_dataset)
            print(
                "Train loss: {:.4f}".format(train_loss / len(train_loader)),
                "Val loss: {:.4f}".format(val_loss / len(test_loader)),
            )
            print(
                "Train accuracy: {:.3f}%".format(train_accuracy),
                "Val accuracy: {:.3f}%".format(val_accuracy),
            )

            if val_accuracy > best_accuracy_val:  # Keep the best model
                best_model = copy.deepcopy(net)
                best_accuracy_val = val_accuracy

    # Save the best model
    torch.save(best_model.state_dict(), MODEL_SAVE_PATH)
