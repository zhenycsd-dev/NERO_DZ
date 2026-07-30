"""
Тестирование модели на новых компонентах
Выполнил: Федоренко Евгений Игоревич, группа ПА-01
"""
import torch
import pickle
import os
import sys

# Добавляем путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем класс MLP из mlp_model.py
from src.mlp_model import MLP

def load_model_and_scaler():
    """Загрузка модели и scaler"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Загрузка модели
    model_path = os.path.join(base_dir, 'models', 'mlp_model.pt')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    model = MLP(
        input_size=checkpoint['input_size'],
        hidden_sizes=checkpoint['hidden_sizes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Загрузка scaler
    scaler_path = os.path.join(base_dir, 'outputs', 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        saved = pickle.load(f)
    
    return model, saved['scaler'], saved['label_encoders']

def predict_component(model, scaler, label_encoders, type_comp, socket, tdp, price):
    """Предсказание для одного компонента"""
    # Преобразование
    type_encoded = label_encoders['Type'].transform([type_comp])[0]
    socket_encoded = label_encoders['Socket'].transform([socket])[0]
    numeric = scaler.transform([[tdp, price]])[0]
    
    # Тензор
    X = torch.tensor([[type_encoded, socket_encoded, numeric[0], numeric[1]]], dtype=torch.float32)
    
    # Предсказание
    with torch.no_grad():
        prob = model(X).item()
        pred = 1 if prob > 0.5 else 0
    
    return prob, pred

def main():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ НА НОВЫХ КОМПОНЕНТАХ")
    print("=" * 60)
    
    # Загрузка
    print("\n1. Загрузка модели и scaler...")
    model, scaler, label_encoders = load_model_and_scaler()
    print("   Модель загружена")
    print(f"   Типы: {label_encoders['Type'].classes_.tolist()}")
    print(f"   Сокеты: {label_encoders['Socket'].classes_.tolist()}")
    
    # Тестовые компоненты
    print("\n2. Тестирование компонентов...")
    
    test_components = [
        {'Type': 'CPU', 'Socket': 'LGA1700', 'TDP': 65, 'Price': 14500, 'expected': 'Совместим'},
        {'Type': 'CPU', 'Socket': 'AM5', 'TDP': 105, 'Price': 22000, 'expected': 'Не совместим'},
        {'Type': 'GPU', 'Socket': 'PCIe x16', 'TDP': 170, 'Price': 31000, 'expected': 'Совместим'},
        {'Type': 'GPU', 'Socket': 'PCIe x16', 'TDP': 263, 'Price': 56000, 'expected': 'Не совместим'},
        {'Type': 'MB', 'Socket': 'AM4', 'TDP': 0, 'Price': 13500, 'expected': 'Совместим'},
        {'Type': 'MB', 'Socket': 'AM5', 'TDP': 0, 'Price': 41000, 'expected': 'Не совместим'},
        {'Type': 'COOLER', 'Socket': 'Universal', 'TDP': 220, 'Price': 3200, 'expected': 'Совместим'},
        {'Type': 'COOLER', 'Socket': 'Universal', 'TDP': 250, 'Price': 11000, 'expected': 'Не совместим'},
    ]
    
    print("\nРезультаты:")
    print("-" * 60)
    print(f"{'Компонент':<25} {'TDP':<8} {'Price':<10} {'Вероятность':<12} {'Результат':<15} {'Ожидание':<15}")
    print("-" * 60)
    
    for comp in test_components:
        prob, pred = predict_component(
            model, scaler, label_encoders,
            comp['Type'], comp['Socket'], comp['TDP'], comp['Price']
        )
        
        result = 'Совместим' if pred == 1 else 'Не совместим'
        status = '✓' if result == comp['expected'] else '✗'
        
        name = f"{comp['Type']} {comp['Socket']}"
        print(f"{name:<25} {comp['TDP']:<8} {comp['Price']:<10} {prob:<12.4f} {result:<15} {comp['expected']:<15} {status}")
    
    print("-" * 60)
    
    # Статистика
    print("\n3. Примеры из датасета (для проверки)...")
    
    dataset_examples = [
        {'Type': 'CPU', 'Socket': 'LGA1700', 'TDP': 58, 'Price': 9000, 'compatible': 1},
        {'Type': 'CPU', 'Socket': 'LGA1700', 'TDP': 253, 'Price': 58000, 'compatible': 0},
        {'Type': 'GPU', 'Socket': 'PCIe x16', 'TDP': 132, 'Price': 26000, 'compatible': 1},
        {'Type': 'GPU', 'Socket': 'PCIe x16', 'TDP': 230, 'Price': 38000, 'compatible': 0},
    ]
    
    print(f"{'Компонент':<25} {'TDP':<8} {'Price':<10} {'Вероятность':<12} {'Предсказание':<15} {'Факт':<10}")
    print("-" * 60)
    
    for ex in dataset_examples:
        prob, pred = predict_component(
            model, scaler, label_encoders,
            ex['Type'], ex['Socket'], ex['TDP'], ex['Price']
        )
        result = 'Совместим' if pred == 1 else 'Не совместим'
        actual = 'Совместим' if ex['compatible'] == 1 else 'Не совместим'
        status = '✓' if pred == ex['compatible'] else '✗'
        
        name = f"{ex['Type']} {ex['Socket']}"
        print(f"{name:<25} {ex['TDP']:<8} {ex['Price']:<10} {prob:<12.4f} {result:<15} {actual:<10} {status}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()