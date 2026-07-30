"""
Умный чат-бот для проверки совместимости компонентов ПК
Выполнил: Федоренко Евгений Игоревич, группа ПА-01
"""
import torch
import torch.nn as nn
import pickle
import os
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

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

class SmartCompatibilityBot:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.is_loaded = False
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.components_db = []
        self.last_search_results = []
        self.last_grouped = {}
        self.last_search_query = ""
        self.load_components_db()
        self.load_model()
    
    def load_components_db(self):
        try:
            csv_path = os.path.join(self.base_dir, 'data', 'components_extended.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(self.base_dir, 'data', 'components.csv')
            df = pd.read_csv(csv_path)
            self.components_db = df.to_dict('records')
            print(f"✅ Загружено {len(self.components_db)} компонентов")
            return True
        except Exception as e:
            self.components_db = []
            print(f"⚠️ База компонентов не загружена: {e}")
            return False
    
    def load_model(self):
        try:
            model_path = os.path.join(self.base_dir, 'models', 'mlp_model.pt')
            if not os.path.exists(model_path):
                model_path = os.path.join(self.base_dir, 'src', 'mlp_model.pt')
            if not os.path.exists(model_path):
                print("⚠️ Модель не найдена")
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
                scaler_path = os.path.join(self.base_dir, 'data', 'scaler.pkl')
            if not os.path.exists(scaler_path):
                print("⚠️ scaler.pkl не найден")
                return False
            with open(scaler_path, 'rb') as f:
                saved = pickle.load(f)
            self.scaler = saved['scaler']
            self.label_encoders = saved['label_encoders']
            self.is_loaded = True
            print("✅ Модель и данные загружены успешно!")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return False
    
    def smart_search(self, query):
        """Умный поиск с приоритетом точных совпадений"""
        results = []
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Определяем тип запроса
        is_cpu_query = any(word in ['ryzen', 'core', 'i3', 'i5', 'i7', 'i9', 'athlon', 'phenom', 'pentium', 'celeron', 'xeon'] for word in words)
        is_gpu_query = any(word in ['gtx', 'rtx', 'radeon', 'rx', 'geforce'] for word in words)
        is_ram_query = any(word in ['ddr', 'kingston', 'corsair', 'g.skill'] for word in words)
        is_socket_query = any(word in ['am4', 'am5', 'lga1700', 'lga1200', 'lga1151', 'lga1150', 'lga1155', 'lga1156', 'lga1366', 'socket', 'pcie', 'ddr2', 'ddr3', 'ddr4', 'ddr5'] for word in words)
        
        # Извлекаем точную модель для CPU
        exact_cpu_model = None
        exact_cpu_series = None
        
        if is_cpu_query:
            # Для Ryzen: ищем "ryzen 5", "ryzen 7" и т.д.
            ryzen_match = re.search(r'ryzen\s+([3579]|1[0-9])', query_lower)
            if ryzen_match:
                exact_cpu_model = f"ryzen {ryzen_match.group(1)}"
                exact_cpu_series = ryzen_match.group(1)
            
            # Для Intel: ищем "i5", "i7" и т.д.
            if not exact_cpu_model:
                intel_match = re.search(r'(i[3579]|i[1-9][0-9])', query_lower)
                if intel_match:
                    exact_cpu_model = intel_match.group(0)
        
        # Для сокетов - формируем точный запрос
        exact_socket = None
        if is_socket_query:
            # Ищем точный сокет
            socket_patterns = ['am4', 'am5', 'lga1700', 'lga1200', 'lga1151', 'lga1150', 'lga1155', 'lga1156', 'lga1366', 'socket 775', 'pcie x16', 'ddr2', 'ddr3', 'ddr4', 'ddr5']
            for pattern in socket_patterns:
                if pattern in query_lower:
                    exact_socket = pattern
                    break
        
        for comp in self.components_db:
            name = str(comp.get('Name', '')).lower()
            comp_type = str(comp.get('Type', '')).lower()
            socket = str(comp.get('Socket', '')).lower()
            
            # Пропускаем служебные строки
            if name.startswith('#') or name.startswith('==='):
                continue
            
            # Если ищем сокет - проверяем только сокет
            if exact_socket:
                if exact_socket in socket or socket == exact_socket:
                    results.append(comp)
                continue
            
            # Фильтрация по типу запроса
            if is_cpu_query and comp_type != 'cpu':
                continue
            if is_gpu_query and comp_type != 'gpu':
                continue
            if is_ram_query and comp_type != 'ram':
                continue
            
            # Для CPU запросов с точной моделью
            if is_cpu_query and exact_cpu_model:
                # Проверяем точное совпадение модели
                if exact_cpu_model in name:
                    results.append(comp)
                    continue
                # Для Ryzen: дополнительная проверка
                if 'ryzen' in query_lower and 'ryzen' in name and exact_cpu_series:
                    # Проверяем, что это точно нужная серия
                    if re.search(f'ryzen\\s+{exact_cpu_series}', name):
                        results.append(comp)
                        continue
                continue  # Пропускаем, если не подходит под точную модель
            
            # Если нет точной модели - обычный поиск
            if query_lower in name:
                results.append(comp)
                continue
            
            # Проверка по отдельным словам (только если не сокет)
            if not exact_socket and len(words) > 1:
                match_count = 0
                for word in words:
                    if word in name:
                        match_count += 2
                    elif word in comp_type or word in socket:
                        match_count += 1
                
                min_matches = len(words) * 0.7
                if match_count >= min_matches:
                    results.append(comp)
        
        # Удаляем дубликаты
        seen = set()
        unique_results = []
        for comp in results:
            comp_id = comp.get('Name', '')
            if comp_id not in seen:
                seen.add(comp_id)
                unique_results.append(comp)
        
        # Финальная фильтрация для Ryzen
        if is_cpu_query and 'ryzen' in query_lower:
            ryzen_series_match = re.search(r'ryzen\s+([3579]|1[0-9])', query_lower)
            if ryzen_series_match:
                target_series = ryzen_series_match.group(1)
                filtered = []
                for comp in unique_results:
                    name = str(comp.get('Name', '')).lower()
                    if 'ryzen' in name:
                        # Проверяем, что серия совпадает
                        if re.search(f'ryzen\\s+{target_series}', name):
                            filtered.append(comp)
                        # Проверяем альтернативный формат: "Ryzen 5 5600X"
                        elif target_series in name and 'ryzen' in name:
                            # Исключаем другие серии
                            if not re.search(r'ryzen\s+[^' + target_series + r']', name):
                                filtered.append(comp)
                    else:
                        filtered.append(comp)
                unique_results = filtered
        
        return unique_results
    
    def predict_component(self, comp):
        comp_type = str(comp.get('Type', ''))
        socket = str(comp.get('Socket', ''))
        tdp = comp.get('TDP', 0)
        price = comp.get('Price', 0)
        if not self.is_loaded:
            return None, "Модель не загружена"
        try:
            if comp_type not in self.label_encoders['Type'].classes_:
                return None, f"Тип '{comp_type}' не найден"
            if socket not in self.label_encoders['Socket'].classes_:
                return None, f"Сокет '{socket}' не найден"
            type_encoded = self.label_encoders['Type'].transform([comp_type])[0]
            socket_encoded = self.label_encoders['Socket'].transform([socket])[0]
            numeric = self.scaler.transform([[tdp, price]])[0]
            X = torch.tensor([[type_encoded, socket_encoded, numeric[0], numeric[1]]], dtype=torch.float32)
            with torch.no_grad():
                prob = self.model(X).item()
                pred = 1 if prob > 0.5 else 0
            return pred, prob
        except Exception as e:
            return None, f"Ошибка: {e}"
    
    def predict_with_params(self, type_comp, socket, tdp, price):
        if not self.is_loaded:
            return "Модель не загружена"
        try:
            if type_comp not in self.label_encoders['Type'].classes_:
                return f"❌ Тип '{type_comp}' не найден. Доступные типы: {self.label_encoders['Type'].classes_.tolist()}"
            if socket not in self.label_encoders['Socket'].classes_:
                return f"❌ Сокет '{socket}' не найден. Доступные сокеты: {self.label_encoders['Socket'].classes_.tolist()}"
            type_encoded = self.label_encoders['Type'].transform([type_comp])[0]
            socket_encoded = self.label_encoders['Socket'].transform([socket])[0]
            numeric = self.scaler.transform([[tdp, price]])[0]
            X = torch.tensor([[type_encoded, socket_encoded, numeric[0], numeric[1]]], dtype=torch.float32)
            with torch.no_grad():
                prob = self.model(X).item()
                pred = 1 if prob > 0.5 else 0
            result = "СОВМЕСТИМ" if pred == 1 else "НЕ СОВМЕСТИМ"
            confidence = prob if pred == 1 else 1 - prob
            return f"""
╔══════════════════════════════════════════════════════════════╗
║                    РЕЗУЛЬТАТ ПРОВЕРКИ                        ║
╠══════════════════════════════════════════════════════════════╣
║  Компонент: {type_comp}                                      ║
║  Сокет:     {socket}                                         ║
║  TDP:       {tdp} Вт                                         ║
║  Цена:      {price} руб.                                     ║
╠══════════════════════════════════════════════════════════════╣
║  Результат: {result}                                         ║
║  Уверенность: {confidence*100:.1f}%                          ║
╚══════════════════════════════════════════════════════════════╝
"""
        except Exception as e:
            return f"❌ Ошибка: {e}"
    
    def show_full_group(self, comp_type):
        """Показать все компоненты определенного типа из последнего поиска"""
        if not self.last_grouped:
            return "❌ Сначала выполните поиск командой predict"
        
        comp_type_lower = comp_type.lower()
        found = False
        for key in self.last_grouped.keys():
            if key.lower() == comp_type_lower:
                comp_type = key
                found = True
                break
        
        if not found:
            available = ", ".join(self.last_grouped.keys())
            return f"❌ Тип '{comp_type}' не найден. Доступны: {available}"
        
        comps = self.last_grouped[comp_type]
        
        output = f"\n📁 {comp_type.upper()} (все {len(comps)} компонентов):\n" + "="*60 + "\n"
        
        for i, comp in enumerate(comps, 1):
            name = str(comp.get('Name', 'Unknown'))
            socket = str(comp.get('Socket', 'N/A'))
            tdp = comp.get('TDP', 0)
            price = comp.get('Price', 0)
            pred, prob = self.predict_component(comp)
            
            output += f"\n{i}. {name}\n"
            output += f"     Сокет: {socket}, TDP: {tdp}Вт, Цена: {price}руб\n"
            if pred is not None:
                result = "✅ СОВМЕСТИМ" if pred == 1 else "❌ НЕ СОВМЕСТИМ"
                output += f"     Совместимость: {result} (уверенность: {prob*100:.1f}%)\n"
        
        return output
    
    def show_search_results(self, query):
        """Показать результаты поиска с группировкой"""
        results = self.smart_search(query)
        
        # Сохраняем результаты
        self.last_search_results = results
        self.last_search_query = query
        
        if not results:
            return f"❌ Компоненты по запросу '{query}' не найдены"
        
        # Фильтруем мусор
        filtered_results = []
        for comp in results:
            name = str(comp.get('Name', ''))
            if name.startswith('#') or name.startswith('==='):
                continue
            if comp.get('TDP', '') == '' or comp.get('Price', '') == '':
                continue
            filtered_results.append(comp)
        
        if not filtered_results:
            return f"❌ По запросу '{query}' найдены только служебные строки"
        
        output = f"\n🔍 Найдено {len(filtered_results)} компонентов по запросу '{query}':\n" + "="*60 + "\n"
        
        # Группируем по типу
        grouped = {}
        for comp in filtered_results:
            comp_type = str(comp.get('Type', 'Unknown'))
            if comp_type not in grouped:
                grouped[comp_type] = []
            grouped[comp_type].append(comp)
        
        self.last_grouped = grouped
        
        for comp_type in sorted(grouped.keys()):
            count = len(grouped[comp_type])
            output += f"\n📁 {comp_type} ({count}):\n"
            for i, comp in enumerate(grouped[comp_type][:5], 1):
                name = str(comp.get('Name', 'Unknown'))
                socket = str(comp.get('Socket', 'N/A'))
                tdp = comp.get('TDP', 0)
                price = comp.get('Price', 0)
                pred, prob = self.predict_component(comp)
                
                output += f"\n  {i}. {name}\n"
                output += f"     Сокет: {socket}, TDP: {tdp}Вт, Цена: {price}руб\n"
                if pred is not None:
                    result = "✅ СОВМЕСТИМ" if pred == 1 else "❌ НЕ СОВМЕСТИМ"
                    output += f"     Совместимость: {result} (уверенность: {prob*100:.1f}%)\n"
            if len(grouped[comp_type]) > 5:
                remaining = len(grouped[comp_type]) - 5
                output += f"\n  ... и еще {remaining} компонентов\n"
                output += f"  💡 Для просмотра всех: more {comp_type}\n"
        
        return output
    
    def show_help(self):
        return """
╔══════════════════════════════════════════════════════════════╗
║                   УМНЫЙ ПОМОЩНИК ПО СБОРКЕ ПК                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   ОСНОВНЫЕ КОМАНДЫ:                                          ║
║  ────────────────────────────────────────────────────────────║
║  predict <название_компонента>                               ║
║    - Поиск и проверка совместимости компонента               ║
║    - Пример: predict ryzen 5                                 ║
║    - Пример: predict gtx 1060                                ║
║    - Пример: predict am4                                     ║
║                                                              ║
║  more <тип>                                                  ║
║    - Показать ВСЕ компоненты указанного типа                 ║
║    - Пример: more CPU                                        ║
║                                                              ║
║  predict <тип> <сокет> <TDP> <цена>                          ║
║    - Проверка с точными параметрами                          ║
║    - Пример: predict CPU LGA1700 65 14500                    ║
║                                                              ║
║  list_types / types                                          ║
║    - Показать все доступные типы компонентов                 ║
║                                                              ║
║  list_sockets / sockets                                      ║
║    - Показать все доступные сокеты                           ║
║                                                              ║
║  help / ?                                                    ║
║    - Показать эту справку                                    ║
║                                                              ║
║  exit / quit                                                 ║
║    - Выйти из чата                                           ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    def show_types(self):
        if self.label_encoders:
            return f"📋 Доступные типы: {', '.join(self.label_encoders['Type'].classes_.tolist())}"
        return "❌ Данные не загружены"
    
    def show_sockets(self):
        if self.label_encoders:
            return f"📋 Доступные сокеты: {', '.join(self.label_encoders['Socket'].classes_.tolist())}"
        return "❌ Данные не загружены"
    
    def run(self):
        """Запуск чат-бота"""
        print("=" * 60)
        print("🧠 УМНЫЙ ЧАТ-БОТ ДЛЯ ПРОВЕРКИ СОВМЕСТИМОСТИ КОМПОНЕНТОВ")
        print("=" * 60)
        
        if not self.is_loaded:
            print("⚠️ Модель не загружена. Попробуйте перезапустить.")
            return
        
        print(self.show_help())
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                cmd = parts[0].lower()
                args = parts[1:]
                
                if cmd in ["exit", "quit", "выход"]:
                    print("\n👋 До свидания!")
                    break
                
                elif cmd in ["help", "?", "помощь"]:
                    print(self.show_help())
                
                elif cmd in ["predict", "p", "предсказать"]:
                    if len(args) >= 1:
                        if len(args) == 4:
                            try:
                                float(args[2])
                                float(args[3])
                                result = self.predict_with_params(
                                    args[0], args[1], float(args[2]), float(args[3])
                                )
                                print(result)
                                continue
                            except ValueError:
                                pass
                        
                        query = " ".join(args)
                        result = self.show_search_results(query)
                        print(result)
                    else:
                        print("❌ Используйте:")
                        print("  predict <название_компонента>")
                        print("  predict <тип> <сокет> <TDP> <цена>")
                        print("")
                        print("Примеры:")
                        print("  predict ryzen 5")
                        print("  predict CPU LGA1700 65 14500")
                        print("  predict am4")
                
                elif cmd in ["more", "m"]:
                    if len(args) != 1:
                        print("❌ Используйте: more <тип>")
                        print("Пример: more CPU")
                        continue
                    result = self.show_full_group(args[0])
                    print(result)
                
                elif cmd in ["list_types", "types", "типы"]:
                    print(self.show_types())
                
                elif cmd in ["list_sockets", "sockets", "сокеты"]:
                    print(self.show_sockets())
                
                else:
                    print(f"❌ Неизвестная команда: {cmd}")
                    print("Введите 'help' для списка команд")
                
            except KeyboardInterrupt:
                print("\n\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")

def main():
    bot = SmartCompatibilityBot()
    bot.run()

if __name__ == "__main__":
    main()