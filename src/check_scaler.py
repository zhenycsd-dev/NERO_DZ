"""
Проверка содержимого scaler.pkl
Выполнил: Федоренко Евгений Игоревич, группа ПА-01
"""
import pickle
import os
import numpy as np

def check_scaler():
    print("=" * 60)
    print("ПРОВЕРКА СОДЕРЖИМОГО scaler.pkl")
    print("=" * 60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    
    try:
        with open(scaler_path, 'rb') as f:
            saved = pickle.load(f)
        
        scaler = saved['scaler']
        label_encoders = saved['label_encoders']
        
        print("\n1. MinMaxScaler:")
        print(f"   feature_range: {scaler.feature_range}")
        print(f"   data_min_: {scaler.data_min_}")
        print(f"   data_max_: {scaler.data_max_}")
        print(f"   data_range_: {scaler.data_range_}")
        print(f"   scale_: {scaler.scale_}")
        
        print("\n2. LabelEncoders:")
        for col_name, encoder in label_encoders.items():
            print(f"   {col_name}:")
            print(f"      classes: {encoder.classes_.tolist()}")
            print(f"      encoding: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
        
        print("\n3. Пример преобразования:")
        # Пример для CPU
        type_encoded = label_encoders['Type'].transform(['CPU'])[0]
        socket_encoded = label_encoders['Socket'].transform(['LGA1700'])[0]
        numeric = scaler.transform([[65, 14500]])[0]
        
        print(f"   CPU -> {type_encoded}")
        print(f"   LGA1700 -> {socket_encoded}")
        print(f"   [65, 14500] -> [{numeric[0]:.4f}, {numeric[1]:.4f}]")
        
        print("\n4. Полная структура scaler.pkl:")
        print("   {")
        print("       'scaler': MinMaxScaler(...),")
        print(f"       'label_encoders': {{")
        print(f"           'Type': {len(label_encoders['Type'].classes_)} классов,")
        print(f"           'Socket': {len(label_encoders['Socket'].classes_)} классов")
        print("       }")
        print("   }")
        
        print("\n" + "=" * 60)
        print("✅ scaler.pkl успешно загружен и проверен!")
        
    except FileNotFoundError:
        print(f"❌ Файл scaler.pkl не найден по пути: {scaler_path}")
        print("   Сначала запустите preprocess.py")
    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")

if __name__ == "__main__":
    check_scaler()