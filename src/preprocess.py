"""
Скрипт предобработки данных
Выполнил: Федоренко Евгений Игоревич, группа ПА-01
"""
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import os

def main():
    print("=" * 60)
    print("ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 60)
    
    # Получаем путь к текущей директории (где лежит preprocess.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nТекущая директория: {current_dir}")
    
    # 1. Чтение данных
    print("\n1. Чтение components.csv...")
    csv_path = os.path.join(current_dir, 'components.csv')
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   Загружено {len(df)} записей")
        print(f"   Колонки: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"   Ошибка: Файл components.csv не найден!")
        return

    # 2. Создание LabelEncoders для категорий
    print("\n2. Применение Label Encoding...")
    label_encoders = {}
    
    # Для Type
    le_type = LabelEncoder()
    df['Type'] = le_type.fit_transform(df['Type'])
    label_encoders['Type'] = le_type
    print(f"   Type: {dict(zip(le_type.classes_, le_type.transform(le_type.classes_)))}")
    
    # Для Socket
    le_socket = LabelEncoder()
    df['Socket'] = le_socket.fit_transform(df['Socket'])
    label_encoders['Socket'] = le_socket
    print(f"   Socket: {dict(zip(le_socket.classes_, le_socket.transform(le_socket.classes_)))}")

    # 3. Нормализация числовых признаков
    print("\n3. Применение MinMaxScaler...")
    scaler = MinMaxScaler()
    numeric_cols = ['TDP', 'Price']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"   TDP: min={scaler.data_min_[0]:.2f}, max={scaler.data_max_[0]:.2f}")
    print(f"   Price: min={scaler.data_min_[1]:.2f}, max={scaler.data_max_[1]:.2f}")

    # 4. Удаление колонки Name
    print("\n4. Удаление колонки Name...")
    df_processed = df.drop('Name', axis=1)
    print(f"   Оставшиеся колонки: {df_processed.columns.tolist()}")

    # 5. Сохранение в тензоры PyTorch
    print("\n5. Конвертация в тензоры PyTorch...")
    X = df_processed.drop('Compatible', axis=1).values
    y = df_processed['Compatible'].values
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    # Сохраняем в ту же папку data
    data_pt_path = os.path.join(current_dir, 'data.pt')
    torch.save({'features': X_tensor, 'labels': y_tensor}, data_pt_path)
    print(f"   data.pt создан (размер: {X_tensor.shape})")

    # 6. Сохранение scaler и label_encoders
    print("\n6. Сохранение scaler.pkl...")
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'label_encoders': label_encoders
        }, f)
    print(f"   scaler.pkl создан")

    print("\n" + "=" * 60)
    print("ПРЕДОБРАБОТКА УСПЕШНО ЗАВЕРШЕНА")
    print("\nСозданы файлы в папке data:")
    print("  data.pt")
    print("  scaler.pkl")
    print("=" * 60)

if __name__ == "__main__":
    main()