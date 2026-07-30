"""
MLP модель для предсказания совместимости компонентов ПК
Выполнил: Федоренко Евгений Игоревич, группа ПА-01
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class MLP(nn.Module):
    """
    Многослойный перцептрон для классификации совместимости
    Архитектура: 4 -> 64 -> 32 -> 16 -> 1
    """
    def __init__(self, input_size=4, hidden_sizes=[64, 32, 16], dropout=0.2):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ MLP МОДЕЛИ")
    print("=" * 60)
    
    # 1. Загрузка данных
    print("\n1. Загрузка данных...")
    data = torch.load('data.pt')
    X = data['features']
    y = data['labels']
    print(f"   Загружено: {X.shape[0]} образцов, {X.shape[1]} признаков")
    
    # 2. Разделение данных
    print("\n2. Разделение данных...")
    X_np = X.numpy()
    y_np = y.numpy().ravel()
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_np, y_np, test_size=0.3, random_state=42, stratify=y_np
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    print(f"   Train: {len(X_train)}")
    print(f"   Val: {len(X_val)}")
    print(f"   Test: {len(X_test)}")
    
    # 3. Создание DataLoader
    print("\n3. Создание DataLoader...")
    batch_size = 8
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    print(f"   Batch size: {batch_size}")
    
    # 4. Создание модели
    print("\n4. Создание модели...")
    model = MLP(input_size=4, hidden_sizes=[64, 32, 16], dropout=0.2)
    print(f"   Архитектура: {model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Параметров: {total_params}")
    
    # 5. Обучение
    print("\n5. Обучение модели...")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"   Устройство: {device}")
    
    epochs = 300
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_accuracies.append(val_acc)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # 6. Оценка на тестовых данных
    print("\n6. Оценка модели...")
    model.eval()
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            test_probs.extend(probs.flatten())
            test_preds.extend(preds.flatten())
            test_labels.extend(batch_y.numpy().flatten())
    
    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)
    
    metrics = {
        'accuracy': accuracy_score(test_labels, test_preds),
        'precision': precision_score(test_labels, test_preds, zero_division=0),
        'recall': recall_score(test_labels, test_preds, zero_division=0),
        'f1': f1_score(test_labels, test_preds, zero_division=0)
    }
    
    print(f"\n   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")
    
    # 7. Матрица ошибок
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\n   Матрица ошибок:")
    print(f"   {cm}")
    
    # 8. Сохранение модели
    print("\n7. Сохранение модели...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': MLP,
        'input_size': 4,
        'hidden_sizes': [64, 32, 16]
    }, 'mlp_model.pt')
    print("   mlp_model.pt сохранён")
    
    # 9. Графики
    print("\n8. Построение графиков...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(val_accuracies, label='Validation Accuracy', linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['Not Compatible', 'Compatible'],
                yticklabels=['Not Compatible', 'Compatible'])
    axes[2].set_title('Confusion Matrix')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    print("   training_results.png сохранён")
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
    print("=" * 60)

if __name__ == "__main__":
    main()