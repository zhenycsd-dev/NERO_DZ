"""
Просмотр предобработанных данных
"""
import torch
import pandas as pd
import numpy as np

def view_processed_data():
    print("=" * 60)
    print("ПРОСМОТР ПРЕДОБРАБОТАННЫХ ДАННЫХ")
    print("=" * 60)
    
    try:
        # Загружаем data.pt
        data = torch.load('data.pt')
        X = data['features']
        y = data['labels']
        
        print(f"\nРазмеры данных:")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")
        
        print(f"\nПервые 5 образцов X:")
        print(X[:5])
        
        print(f"\nПервые 5 меток y:")
        print(y[:5].flatten())
        
        print(f"\nСтатистика по X:")
        print(f"  Min: {X.min(dim=0)[0]}")
        print(f"  Max: {X.max(dim=0)[0]}")
        print(f"  Mean: {X.mean(dim=0)}")
        print(f"  Std: {X.std(dim=0)}")
        
        print(f"\nРаспределение меток:")
        unique, counts = torch.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Класс {int(u)}: {int(c)} образцов ({int(c)/len(y)*100:.1f}%)")
            
    except FileNotFoundError:
        print("❌ Файл data.pt не найден!")
        print("   Сначала запустите preprocess.py")

if __name__ == "__main__":
    view_processed_data()