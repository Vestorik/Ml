from torchvision import datasets, transforms
from config import BASE_PATH
import torch as tr


BATCH_SIZE = 100

# Используем набор данных CIFAR10

# Получаем данные
path = BASE_PATH / "data"
path.mkdir(parents=True, exist_ok=True)

# Преобразования с нормализацией
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(
    root=path, train=True, transform=transform, download=True
)
test_dataset = datasets.CIFAR10(
    root=path, train=False, transform=transform, download=True
)

#    Подготовка данных
train_loader = tr.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = tr.utils.data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
)
