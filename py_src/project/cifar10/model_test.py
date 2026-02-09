"""
Модуль для тестирования модели
"""


import torch as tr


def testing_model(test_loader, model, DEVICE):
    """
    Оценивает производительность обученной модели на тестовом наборе данных.

    Функция переводит модель в режим оценки (eval), проходит по всем пакетам тестовых данных,
    вычисляет точность предсказаний, среднюю уверенность модели в правильных предсказаниях,
    а также взвешенную оценку, сочетающую точность и уверенность.

    Параметры:
        test_loader (DataLoader): Загрузчик тестовых данных, содержащий пары (изображение, метка).
        model (nn.Module): Обученная модель PyTorch, которую необходимо протестировать.
        DEVICE (torch.device): Устройство ('cpu' или 'cuda'), на котором выполняются вычисления.

    Возвращает:
        float: Взвешенная оценка модели, равная произведению общей точности
               на среднюю уверенность в правильных предсказаниях.

    Примечания:
        - Модель переводится в режим `eval`, чтобы отключить слои, зависящие от режима (например, Dropout, BatchNorm).
        - Вычисления выполняются без подсчёта градиентов для экономии памяти и ускорения.
        - Уверенность определяется как максимальное значение softmax по выходам модели.
        - Средняя уверенность рассчитывается только по тем предсказаниям, которые оказались верными.
    """
    model.eval()  # Переводим модель в режим оценки
    
    #  Статистические переменные
    total_correct = 0
    total_samples = 0
    total_confidence = 0.0

    with tr.no_grad(): # Отключаем расчёт градтентов
        for images, labels in test_loader: 
            images, labels = images.to(DEVICE), labels.to(DEVICE) # переносим данные на DEVICE

            outputs = model(images)  # Получаем предсказание
            probs = tr.softmax(outputs, dim=1)           # вероятности
            max_probs, predicted = tr.max(probs, dim=1)  # уверенность и класс

            # Подсчёт обычной точности
            correct = (predicted == labels)
            total_correct += correct.sum().item()
            total_samples += labels.size(0)

            # Средняя уверенность в правильных ответах
            total_confidence += max_probs[correct].sum().item()

    overall_accuracy = total_correct / total_samples
    avg_confidence_in_correct = total_confidence / total_correct

    # Взвешенная оценка: "качество модели"
    weighted_score = overall_accuracy * avg_confidence_in_correct

    print(f"Общая точность: {overall_accuracy:.3f}")
    print(f"Средняя уверенность в правильных ответах: {avg_confidence_in_correct:.3f}")
    print(f"Взвешенная оценка: {weighted_score:.3f}")
    return weighted_score