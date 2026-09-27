import torch
import torch.nn as nn
import torch.optim as optim


def calculate_accuracy(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            predictions = torch.argmax(model(input_ids), dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples * 100


def train(model, train_loader, val_loader, device, epochs=3, lr=3e-4, log_every=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss = criterion(model(input_ids), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % log_every == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], "
                      f"Loss: {running_loss/log_every:.4f}")
                running_loss = 0.0

        val_accuracy = calculate_accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.2f}%")
