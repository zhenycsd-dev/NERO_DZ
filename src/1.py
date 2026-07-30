"""
Ультимативный чат-бот-консультант по сборке ПК
Поддерживает все поколения компонентов: от винтажных до современных
Выполнил: Федоренко Евгений Игоревич, группа ПА-01
"""
import torch
import torch.nn as nn
import pickle
import os
import json
from datetime import datetime
import re
import math
import pandas as pd

# Класс MLP (нейросеть для предсказания совместимости)
class MLP(nn.Module):
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

class PcCompatibilityBot:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.is_loaded = False
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.components_db = []
        
        # Сборка пользователя
        self.build = {
            'cpu': None,
            'motherboard': None,
            'gpu': None,
            'ram': None,
            'storage': None,
            'cooler': None,
            'psu': None
        }
        
        # База знаний о совместимости
        self.socket_compatibility = {
            'Socket 775': ['DDR2', 'DDR3'],
            'AM2': ['DDR2'],
            'AM3': ['DDR3'],
            'AM3+': ['DDR3'],
            'LGA1156': ['DDR3'],
            'LGA1155': ['DDR3'],
            'LGA1150': ['DDR3'],
            'LGA1151': ['DDR4'],
            'LGA1200': ['DDR4'],
            'LGA1700': ['DDR4', 'DDR5'],
            'AM4': ['DDR4'],
            'AM5': ['DDR5']
        }
        
        self.ddr_voltage = {
            'DDR2': 1.8,
            'DDR3': 1.5,
            'DDR4': 1.2,
            'DDR5': 1.1
        }
        
        self.ddr_speed = {
            'DDR2': '400-1066 MHz',
            'DDR3': '800-2133 MHz',
            'DDR4': '2133-4800 MHz',
            'DDR5': '4800-8000 MHz'
        }
        
        # Рекомендации по блоку питания (в Ваттах)
        self.psu_recommendations = {
            'retro': 300,        # Винтажные ПК
            'office': 400,       # Офисные
            'gaming_entry': 550, # Начальный игровой
            'gaming_mid': 650,   # Средний игровой
            'gaming_high': 750,  # Высокий игровой
            'workstation': 850,  # Рабочая станция
            'extreme': 1000      # Экстрим
        }
        
        # Загрузка данных
        self.load_components_db()
        self.load_model()
    
    def load_components_db(self):
        """Загрузка базы компонентов из CSV"""
        try:
            # Пробуем загрузить расширенную базу
            csv_path = os.path.join(self.base_dir, 'data', 'components_extended.csv')
            if not os.path.exists(csv_path):
                # Если расширенной нет, используем обычную
                csv_path = os.path.join(self.base_dir, 'data', 'components.csv')
            
            df = pd.read_csv(csv_path)
            self.components_db = df.to_dict('records')
            
            # Подсчет компонентов по типам
            type_counts = {}
            for comp in self.components_db:
                t = comp.get('Type', 'Unknown')
                type_counts[t] = type_counts.get(t, 0) + 1
            
            print(f"✅ Загружено {len(self.components_db)} компонентов:")
            for t, count in type_counts.items():
                print(f"   {t}: {count}")
            return True
        except Exception as e:
            self.components_db = []
            print(f"⚠️ База компонентов не загружена: {e}")
            return False
    
    def load_model(self):
        """Загрузка модели и scaler"""
        try:
            model_path = os.path.join(self.base_dir, 'models', 'mlp_model.pt')
            if not os.path.exists(model_path):
                print("⚠️ Модель не найдена. Сначала обучите модель: python mlp_model.py")
                return False
            
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            
            self.model = MLP(
                input_size=checkpoint.get('input_size', 4),
                hidden_sizes=checkpoint.get('hidden_sizes', [64, 32, 16])
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            scaler_path = os.path.join(self.base_dir, 'outputs', 'scaler.pkl')
            if not os.path.exists(scaler_path):
                print("⚠️ scaler.pkl не найден. Запустите preprocess.py")
                return False
            
            with open(scaler_path, 'rb') as f:
                saved = pickle.load(f)
            
            self.scaler = saved['scaler']
            self.label_encoders = saved['label_encoders']
            
            self.is_loaded = True
            print("✅ Модель загружена успешно!")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def predict_compatibility(self, type_comp, socket, tdp, price):
        """Предсказание совместимости компонента"""
        if not self.is_loaded:
            return None, "Модель не загружена"
        
        try:
            if type_comp not in self.label_encoders['Type'].classes_:
                return None, f"❌ Тип '{type_comp}' не найден. Доступны: {self.label_encoders['Type'].classes_.tolist()}"
            if socket not in self.label_encoders['Socket'].classes_:
                return None, f"❌ Сокет '{socket}' не найден. Доступны: {self.label_encoders['Socket'].classes_.tolist()}"
            
            type_encoded = self.label_encoders['Type'].transform([type_comp])[0]
            socket_encoded = self.label_encoders['Socket'].transform([socket])[0]
            numeric = self.scaler.transform([[tdp, price]])[0]
            
            X = torch.tensor([[type_encoded, socket_encoded, numeric[0], numeric[1]]], dtype=torch.float32)
            
            with torch.no_grad():
                prob = self.model(X).item()
                pred = 1 if prob > 0.5 else 0
            
            return pred, prob
        except Exception as e:
            return None, f"❌ Ошибка: {e}"
    
    def check_motherboard_compatibility(self, cpu_socket, ram_type, storage_type):
        """Проверка совместимости материнской платы"""
        issues = []
        warnings = []
        
        # Проверка сокета и памяти
        if cpu_socket in self.socket_compatibility:
            if ram_type not in self.socket_compatibility[cpu_socket]:
                issues.append(f"❌ {ram_type} не поддерживается для сокета {cpu_socket}")
        else:
            warnings.append(f"⚠️ Неизвестный сокет: {cpu_socket}")
        
        # Проверка хранилища
        if storage_type == 'M.2 NVMe' and cpu_socket in ['AM4', 'LGA1151']:
            warnings.append("⚠️ NVMe может работать с ограничениями (проверьте конкретную модель материнской платы)")
        
        return issues, warnings
    
    def calculate_psu_requirement(self):
        """Расчет необходимой мощности блока питания"""
        total_tdp = 0
        components_power = {
            'cpu': 0,
            'gpu': 0,
            'ram': 0,
            'storage': 0,
            'cooler': 0,
            'motherboard': 30  # базовое потребление материнской платы
        }
        
        # Сбор TDP всех компонентов
        for key, comp in self.build.items():
            if comp and 'TDP' in comp:
                if key == 'cpu':
                    components_power['cpu'] = comp['TDP']
                elif key == 'gpu':
                    components_power['gpu'] = comp['TDP']
                elif key == 'ram':
                    components_power['ram'] = comp.get('TDP', 5)
                elif key == 'storage':
                    components_power['storage'] = comp.get('TDP', 5)
                elif key == 'cooler':
                    components_power['cooler'] = comp.get('TDP', 5)
        
        # Суммируем
        for key, power in components_power.items():
            total_tdp += power
        
        # Добавляем запас 30% на пиковые нагрузки
        recommended_psu = math.ceil(total_tdp * 1.3 / 50) * 50  # округление до 50
        
        # Дополнительная рекомендация для старых систем
        if total_tdp < 150:
            recommended_psu = max(recommended_psu, 300)
        
        return total_tdp, recommended_psu
    
    def get_psu_category(self, power):
        """Определение категории блока питания по мощности"""
        if power < 400:
            return 'retro'
        elif power < 550:
            return 'office'
        elif power < 650:
            return 'gaming_entry'
        elif power < 750:
            return 'gaming_mid'
        elif power < 850:
            return 'gaming_high'
        elif power < 1000:
            return 'workstation'
        else:
            return 'extreme'
    
    def analyze_build(self):
        """Полный анализ сборки"""
        if not any(self.build.values()):
            return "📭 Сборка пуста. Добавьте компоненты командой add"
        
        report = []
        report.append("="*60)
        report.append("🔍 ПОЛНЫЙ АНАЛИЗ СБОРКИ")
        report.append("="*60)
        
        # Список компонентов
        report.append("\n📋 КОМПОНЕНТЫ:")
        for key, comp in self.build.items():
            if comp:
                name = comp.get('Name', comp.get('Type', 'Unknown'))
                tdp = comp.get('TDP', 0)
                report.append(f"  {key.upper()}: {name} ({tdp}Вт)")
            else:
                report.append(f"  {key.upper()}: ❌ Не выбран")
        
        # Проверка совместимости CPU и материнской платы
        report.append("\n🔗 ПРОВЕРКА СОВМЕСТИМОСТИ:")
        
        if self.build['cpu'] and self.build['motherboard']:
            cpu_socket = self.build['cpu'].get('Socket', 'Unknown')
            mb_socket = self.build['motherboard'].get('Socket', 'Unknown')
            
            if cpu_socket == mb_socket:
                report.append(f"  ✅ Сокет CPU ({cpu_socket}) совместим с материнской платой")
            else:
                report.append(f"  ❌ Сокет CPU ({cpu_socket}) НЕ совместим с материнской платой ({mb_socket})")
        
        # Проверка памяти
        if self.build['ram']:
            ram_type = self.build['ram'].get('Socket', 'Unknown')
            if self.build['motherboard']:
                mb_name = self.build['motherboard'].get('Name', '')
                if 'DDR5' in mb_name and ram_type == 'DDR4':
                    report.append("  ⚠️ Материнская плата с DDR5, а память DDR4")
                elif 'DDR4' in mb_name and ram_type == 'DDR5':
                    report.append("  ⚠️ Материнская плата с DDR4, а память DDR5")
                elif ram_type in ['DDR2', 'DDR3', 'DDR4', 'DDR5']:
                    report.append(f"  ✅ Память {ram_type} поддерживается")
            else:
                report.append(f"  ℹ️ Память {ram_type} (выберите материнскую плату для проверки)")
        
        # Проверка накопителя
        if self.build['storage']:
            storage_type = self.build['storage'].get('Socket', 'Unknown')
            if storage_type in ['M.2 NVMe', 'M.2 SATA']:
                report.append(f"  ✅ Накопитель {storage_type} (проверьте поддержку на материнской плате)")
            else:
                report.append(f"  ℹ️ Накопитель {storage_type}")
        
        # Расчет блока питания
        total_tdp, recommended_psu = self.calculate_psu_requirement()
        category = self.get_psu_category(recommended_psu)
        
        report.append(f"\n⚡ БЛОК ПИТАНИЯ:")
        report.append(f"  Суммарное TDP: {total_tdp} Вт")
        report.append(f"  Рекомендуемая мощность: {recommended_psu} Вт")
        report.append(f"  Категория: {category} ({self.psu_recommendations.get(category, 0)}Вт)")
        
        if self.build['psu']:
            psu_power = self.build['psu'].get('TDP', 0) or 0
            psu_name = self.build['psu'].get('Name', 'Unknown')
            
            if psu_power == 0:
                # Если в базе нет TDP для PSU, ищем по названию
                import re
                power_match = re.search(r'(\d+)(?:\s*[Ww])', psu_name)
                if power_match:
                    psu_power = int(power_match.group(1))
                else:
                    psu_power = 500  # значение по умолчанию
            
            if psu_power >= recommended_psu:
                report.append(f"  ✅ БП {psu_power} Вт достаточен (запас {psu_power - recommended_psu} Вт)")
            else:
                report.append(f"  ⚠️ БП {psu_power} Вт может быть недостаточен (нужно {recommended_psu} Вт)")
                report.append(f"     Рекомендация: замените на {recommended_psu}+ Вт")
        else:
            report.append(f"  💡 Рекомендуемый БП: {recommended_psu} Вт или выше")
        
        # Общая оценка
        report.append("\n📊 ОБЩАЯ ОЦЕНКА:")
        issues = 0
        if self.build['cpu'] and self.build['motherboard']:
            if cpu_socket != mb_socket:
                issues += 1
        
        if issues == 0:
            report.append("  ✅ Сборка выглядит совместимой!")
        else:
            report.append("  ⚠️ Есть проблемы с совместимостью (смотрите выше)")
        
        report.append("="*60)
        return "\n".join(report)
    
    def add_component(self, component_type, search_query):
        """Добавление компонента в сборку"""
        # Поиск в базе
        matches = []
        for comp in self.components_db:
            if comp.get('Type', '').upper() == component_type.upper():
                name = comp.get('Name', '').lower()
                if search_query.lower() in name:
                    matches.append(comp)
        
        if not matches:
            return f"❌ Компонент '{search_query}' не найден для типа {component_type}"
        
        if len(matches) > 1:
            result = f"🔍 Найдено {len(matches)} компонентов:\n"
            for i, comp in enumerate(matches[:10], 1):
                name = comp.get('Name', 'Unknown')
                socket = comp.get('Socket', 'N/A')
                tdp = comp.get('TDP', 0)
                price = comp.get('Price', 0)
                result += f"  {i}. {name} ({socket}, {tdp}Вт, {price}руб)\n"
            if len(matches) > 10:
                result += f"  ... и еще {len(matches)-10} компонентов\n"
            result += "Используйте номер для выбора: add <тип> <номер>"
            return result
        
        # Добавление компонента
        comp = matches[0]
        comp_type_lower = component_type.lower()
        
        # Маппинг типов на ключи сборки
        type_mapping = {
            'cpu': 'cpu',
            'processor': 'cpu',
            'mb': 'motherboard',
            'motherboard': 'motherboard',
            'gpu': 'gpu',
            'videocard': 'gpu',
            'ram': 'ram',
            'memory': 'ram',
            'storage': 'storage',
            'ssd': 'storage',
            'm2': 'storage',
            'cooler': 'cooler',
            'cooling': 'cooler',
            'psu': 'psu',
            'power': 'psu'
        }
        
        key = type_mapping.get(comp_type_lower)
        if key:
            self.build[key] = comp
            return f"✅ Добавлен: {comp.get('Name', 'Unknown')}"
        else:
            return f"❌ Неизвестный тип: {component_type}"
    
    def show_build_status(self):
        """Показать статус сборки"""
        if not any(self.build.values()):
            return "📭 Сборка пуста"
        
        status = "\n🖥️ СТАТУС СБОРКИ:\n" + "-"*50 + "\n"
        
        total_tdp = 0
        total_price = 0
        
        for key, comp in self.build.items():
            if comp:
                name = comp.get('Name', 'Unknown')
                tdp = comp.get('TDP', 0)
                price = comp.get('Price', 0)
                socket = comp.get('Socket', 'N/A')
                status += f"  {key.upper()}: {name}\n"
                status += f"      Сокет: {socket}, TDP: {tdp}Вт, Цена: {price}руб\n"
                total_tdp += tdp
                total_price += price
            else:
                status += f"  {key.upper()}: ❌ Не выбран\n"
        
        status += f"\n📊 ИТОГО: TDP {total_tdp}Вт, Цена {total_price}руб"
        
        return status
    
    def find_components(self, query):
        """Поиск компонентов по названию"""
        results = []
        for comp in self.components_db:
            name = comp.get('Name', '').lower()
            if query.lower() in name:
                results.append(comp)
        
        if not results:
            return f"❌ Компоненты по запросу '{query}' не найдены"
        
        result = f"🔍 Найдено {len(results)} компонентов:\n"
        for comp in results[:10]:
            name = comp.get('Name', 'Unknown')
            comp_type = comp.get('Type', 'Unknown')
            socket = comp.get('Socket', 'N/A')
            tdp = comp.get('TDP', 0)
            price = comp.get('Price', 0)
            result += f"  • {name} ({comp_type}, {socket}, {tdp}Вт, {price}руб)\n"
        
        if len(results) > 10:
            result += f"  ... и еще {len(results)-10} компонентов"
        
        return result
    
    def search_components(self, comp_type, socket=None, max_tdp=None, max_price=None):
        """Поиск компонентов по параметрам"""
        results = []
        for comp in self.components_db:
            if comp.get('Type', '').upper() != comp_type.upper():
                continue
            if socket and comp.get('Socket', '').upper() != socket.upper():
                continue
            if max_tdp and comp.get('TDP', 0) > max_tdp:
                continue
            if max_price and comp.get('Price', 0) > max_price:
                continue
            results.append(comp)
        
        if not results:
            return f"❌ Компоненты типа {comp_type} с заданными параметрами не найдены"
        
        result = f"🔍 Найдено {len(results)} компонентов типа {comp_type}:\n"
        for comp in results[:10]:
            name = comp.get('Name', 'Unknown')
            socket = comp.get('Socket', 'N/A')
            tdp = comp.get('TDP', 0)
            price = comp.get('Price', 0)
            result += f"  • {name} ({socket}, {tdp}Вт, {price}руб)\n"
        
        if len(results) > 10:
            result += f"  ... и еще {len(results)-10} компонентов"
        
        return result
    
    def clear_build(self):
        """Очистка сборки"""
        self.build = {k: None for k in self.build}
        return "🧹 Сборка очищена"
    
    def show_help(self):
        """Показать расширенную справку"""
        return """
╔══════════════════════════════════════════════════════════════╗
║           УЛЬТИМАТИВНЫЙ КОНСУЛЬТАНТ ПО СБОРКЕ ПК             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   ОСНОВНЫЕ КОМАНДЫ:                                          ║
║  ────────────────────────────────────────────────────────────║
║  predict <тип> <сокет> <TDP> <цена>                          ║
║    - Проверить совместимость компонента                      ║
║    - Пример: predict CPU LGA1700 65 14500                    ║
║                                                              ║
║  add <тип> <название или номер>                              ║
║    - Добавить компонент в сборку                             ║
║    - Типы: cpu, mb, gpu, ram, storage, cooler, psu           ║
║    - Пример: add cpu i5, add ram 2                           ║
║                                                              ║
║  build / b                                                   ║
║    - Показать текущую сборку                                 ║
║                                                              ║
║  analyze / an                                                ║
║    - Полный анализ совместимости сборки                      ║
║                                                              ║
║  clear / c                                                   ║
║    - Очистить сборку                                         ║
║                                                              ║
║  find <запрос>                                               ║
║    - Поиск компонентов по названию                           ║
║    - Пример: find pentium, find ddr4                         ║
║                                                              ║
║  search <тип> [сокет] [макс TDP] [макс цена]                 ║
║    - Поиск компонентов по параметрам                         ║
║    - Пример: search CPU LGA1700 200 50000                    ║
║    - Пример: search GPU                                      ║
║                                                              ║
║  list_types / types                                          ║
║    - Показать доступные типы компонентов                     ║
║                                                              ║
║  list_sockets / sockets                                      ║
║    - Показать доступные сокеты                               ║
║                                                              ║
║  help / ?                                                    ║
║    - Показать эту справку                                    ║
║                                                              ║
║  exit / quit                                                 ║
║    - Выйти из чата                                           ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    def run(self):
        """Запуск чат-бота"""
        print("="*60)
        print("🖥️  УЛЬТИМАТИВНЫЙ КОНСУЛЬТАНТ ПО СБОРКЕ ПК")
        print("="*60)
        print("\nВведите 'help' для списка команд\n")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()
                args = cmd_parts[1:]
                
                if cmd in ['exit', 'quit']:
                    print("\n👋 До свидания!")
                    break
                
                elif cmd in ['help', '?']:
                    print(self.show_help())
                
                elif cmd in ['predict', 'p']:
                    if len(args) != 4:
                        print("❌ Нужно 4 аргумента: predict <тип> <сокет> <TDP> <цена>")
                        continue
                    try:
                        pred, prob = self.predict_compatibility(
                            args[0], args[1], float(args[2]), float(args[3])
                        )
                        if pred is None:
                            print(prob)
                        else:
                            result = "✅ СОВМЕСТИМ" if pred == 1 else "❌ НЕ СОВМЕСТИМ"
                            print(f"\n📊 Результат: {result}")
                            print(f"🎯 Уверенность: {prob*100:.1f}%")
                    except ValueError:
                        print("❌ TDP и цена должны быть числами")
                
                elif cmd in ['add', 'a']:
                    if len(args) < 2:
                        print("❌ Нужно: add <тип> <название или номер>")
                        print("Пример: add CPU i5")
                        continue
                    
                    # Проверяем, является ли аргумент номером
                    if args[1].isdigit():
                        # Поиск по номеру из предыдущего запроса
                        # (упрощенная версия - используем последние результаты поиска)
                        print("ℹ️ Используйте полное название компонента")
                        continue
                    
                    result = self.add_component(args[0], ' '.join(args[1:]))
                    print(result)
                
                elif cmd in ['build', 'b']:
                    print(self.show_build_status())
                
                elif cmd in ['analyze', 'an']:
                    print(self.analyze_build())
                
                elif cmd in ['clear', 'c']:
                    print(self.clear_build())
                
                elif cmd in ['find', 'f']:
                    if not args:
                        print("❌ Введите запрос для поиска")
                        continue
                    print(self.find_components(' '.join(args)))
                
                elif cmd in ['search', 's']:
                    if not args:
                        print("❌ Нужно: search <тип> [сокет] [макс TDP] [макс цена]")
                        continue
                    
                    comp_type = args[0]
                    socket = args[1] if len(args) > 1 else None
                    max_tdp = float(args[2]) if len(args) > 2 else None
                    max_price = float(args[3]) if len(args) > 3 else None
                    
                    result = self.search_components(comp_type, socket, max_tdp, max_price)
                    print(result)
                
                elif cmd in ['list_types', 'types']:
                    if self.label_encoders:
                        types = self.label_encoders['Type'].classes_.tolist()
                        print(f"📋 Доступные типы: {', '.join(types)}")
                    else:
                        print("❌ Данные не загружены")
                
                elif cmd in ['list_sockets', 'sockets']:
                    if self.label_encoders:
                        sockets = self.label_encoders['Socket'].classes_.tolist()
                        print(f"📋 Доступные сокеты: {', '.join(sockets)}")
                    else:
                        print("❌ Данные не загружены")
                
                else:
                    print(f"❌ Неизвестная команда: {cmd}")
                    print("Введите 'help' для списка команд")
                
            except KeyboardInterrupt:
                print("\n\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")

def main():
    bot = PcCompatibilityBot()
    bot.run()

if __name__ == "__main__":
    main()