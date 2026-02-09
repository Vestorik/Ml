from torch import device
import torch as tr


def training_model(num_epoch: int, train_data, model, optimaizer, loss_func, device: device):
    """
    Обучает модель на заданном наборе данных в течение указанного количества эпох.

    Функция выполняет обучение модели с использованием переданного оптимизатора и функции потерь.
    На каждой эпохе вычисляются средняя ошибка (loss) и точность (accuracy) на обучающих данных,
    которые выводятся в стандартный поток. Модель переводится в режим обучения (train).

    Параметры:
        num_epoch (int): Количество эпох обучения.
        train_data (DataLoader): Загрузчик обучающих данных, предоставляющий батчи в формате (изображения, метки).
        model (nn.Module): Обучаемая модель PyTorch.
        optimaizer (Optimizer): Оптимизатор (например, SGD, Adam), используемый для обновления весов модели.
        loss_func (Callable): Функция потерь, используемая для вычисления ошибки (например, CrossEntropyLoss).
        device (torch.device): Устройство, на котором выполняются вычисления ('cpu' или 'cuda').

    Возвращает:
        None: Результаты выводятся в консоль; функция не возвращает значений.

    Примечания:
        - Модель переводится в режим `train()` для активации слоёв, зависящих от режима (Dropout, BatchNorm).
        - Используется ручной цикл обучения с накоплением общего loss и подсчётом точности.
        - Точность вычисляется как доля правильно предсказанных меток среди всех образцов.
        - Средний loss за эпоху — это сумма loss по батчам, делённая на количество батчей.
        - Внутри эпохи отображается прогресс в виде строки с заменой предыдущего вывода (используется `\r`).
        - Требует корректной передачи всех аргументов; ожидается, что `train_data` поддерживает `len()`.

    Исключения:
        Может вызвать RuntimeError, если данные или модель находятся на разных устройствах.
        TypeError, если `train_data` не поддерживает `len()` или итерацию.

    Пример использования:
        >>> from torch.optim import Adam
        >>> from torch.nn import CrossEntropyLoss
        >>> optimizer = Adam(model.parameters(), lr=0.001)
        >>> criterion = CrossEntropyLoss()
        >>> training_model(10, train_loader, model, optimizer, criterion, device)
    """
    model.train()  # Переводим модель в режим обучения

    print(f"Эпохи {num_epoch}\n  № Эпохи | Ошибка | Точность\n")
    
    for epoch in range(num_epoch): # Цикл по эпохам



        running_loss = 0.0 # общая ошибка
        correct_train = 0 # правильных ответов
        total_train = 0 # всего тренировок (проейденых выборок)


        for idx, (images, labels) in enumerate(train_data): # Цикл по данным
            # Переносим данные на 
            images, labels = images.to(device), labels.to(device)

            # Обнуляем градиенты
            optimaizer.zero_grad()
            
            #  Получаем предсказания модели
            outputs = model(images)
            
            #  Вычисляем отклонения
            loss = loss_func(outputs, labels)
            
            #  Совершаем обратный проход и обновляем веса
            loss.backward()
            optimaizer.step()

            # 
            running_loss += loss.item()

            # Подсчёт точности
            _, predicted = tr.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            print(f"Эпоха: {epoch + 1}/{num_epoch} {idx / len(train_data):.2f} % | Ошибка: {loss.item():.4f}", end="\r")

        train_acc = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_data)

        print(f"{epoch+1:^9} | {avg_loss:^6.4f} | {train_acc:^6.2f}%")