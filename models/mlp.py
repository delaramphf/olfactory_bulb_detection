import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_mlp(X, Y, hidden_size=100, epochs=10, batch_size=32, learning_rate=0.001, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=random_state)
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MLP(X.shape[1], hidden_size, len(np.unique(Y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model, X_train, X_test, y_train, y_test

def evaluate_mlp(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
    
    y_pred = predicted.numpy()
    y_true = y_test.numpy()
    
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    
    return accuracy, conf_matrix, class_report

def mlp_pipeline(X, Y, hidden_size=100, epochs=10, batch_size=32, learning_rate=0.001, random_state=0):
    model, X_train, X_test, y_train, y_test = train_mlp(X, Y, hidden_size, epochs, batch_size, learning_rate, random_state)
    accuracy, conf_matrix, class_report = evaluate_mlp(model, X_test, y_test)
    
    return {
        'classifier': model,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'X_train': X_train.numpy(),
        'X_test': X_test.numpy(),
        'y_train': y_train.numpy(),
        'y_test': y_test.numpy()
    }
