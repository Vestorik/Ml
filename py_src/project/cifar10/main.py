"""
Обучаем и тестируем модель для класификации изображений CIFAR-10

Получение данных, тренировка и тестирование модели релизованы в отдельных файлах

Используем архитектуру CNN
"""

import torch as tr
from net.CNN import CNN
from data_work import train_loader, test_loader
from model_train import training_model
from model_test import testing_model

# Устанавливаем гиперпараметры
# BATCH_SIZE = 100   // перенесено в train_model.py
INPUT_SIZE = 32 * 32  # входные данные тензор 3 по 32*32
NUM_CLASSES = 10  # 10 классов
NUM_EPOCH = 10  # 10 эпох ~ 75% точности далее рост замедляется
LEARN_RATE = 0.001
DEVICE = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
LEARN = True  # Обучаем модель или пытаемся использовать предобученную

print(DEVICE)
# Определяем модель и переносим её на DEVICE (предположительно GPU)
model = CNN(NUM_CLASSES, INPUT_SIZE)
model = model.to(DEVICE)


if LEARN:
    # Определяем оптимизатор и функцию потерь
    optimaizer = tr.optim.Adam(model.parameters(), lr=LEARN_RATE)
    loss_func = tr.nn.CrossEntropyLoss()

    # Тренируем модель
    training_model(NUM_EPOCH, train_loader, model, optimaizer, loss_func, DEVICE)
else:

    # Загружаем веса модели
    model.load_state_dict(tr.load("cnn_model.pth"))

# Тестируем модель
result = testing_model(test_loader, model, DEVICE)
