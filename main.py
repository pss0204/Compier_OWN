import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from torchviz import make_dot
import model

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

model_instance = model.Model()
# 모델 시각화
dummy_input = torch.randn(1, 1, 28, 28)
output = model_instance(dummy_input)
dot = make_dot(output, params=dict(model_instance.named_parameters()))
dot.render("model_structure", format="png")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_instance.parameters(), lr=0.001)

# 학습 루프
for epoch in range(10):
    model_instance.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_instance(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1} 완료')

# 평가 함수
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(cm)

# 테스트
evaluate(model_instance, test_loader)