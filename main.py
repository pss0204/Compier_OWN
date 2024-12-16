import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from torchviz import make_dot
import model
import time
import argparse

# main.py 시작 부분에 device 설정 추가
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 데이터 변환
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 데이터 로드
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train(model_instance, train_loader, criterion, optimizer, epochs=10):
    model_instance.to(device)#using device
    for epoch in range(epochs):
        model_instance.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)#move to device
            optimizer.zero_grad()
            outputs = model_instance(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} 완료')
    # 모델 저장
    torch.save(model_instance.state_dict(), 'model.pth')
    print('모델 저장 완료')

def evaluate(model_instance, dataloader):
    model_instance.to(device)#using device
    model_instance.eval()
    all_preds, all_labels = [], []
    inference_times = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs=inputs.to(device)#move to device
            start_time = time.time()
            outputs = model_instance(inputs)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f'Average Inference Time per Batch: {average_inference_time:.6f} seconds')
    
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(cm)

def main():
    parser = argparse.ArgumentParser(description='Train or Evaluate the Model')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True, help='Mode: train or evaluate')
    args = parser.parse_args()

    model_instance = model.Model()

    if args.mode == 'train':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_instance.parameters(), lr=0.001)
        train(model_instance, train_loader, criterion, optimizer)
    elif args.mode == 'evaluate':
        # 모델 로드
        model_instance.load_state_dict(torch.load('model.pth'))
        # 모델 시각화 (옵션)
        dummy_input = torch.randn(1, 1, 28, 28)
        output = model_instance(dummy_input)
        dot = make_dot(output, params=dict(model_instance.named_parameters()))
        dot.render("model_structure", format="png")
        evaluate(model_instance, test_loader)

if __name__ == '__main__':
    main()