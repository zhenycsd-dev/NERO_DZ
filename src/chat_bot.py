"""
Умный чат-бот для проверки совместимости компонентов ПК
С поддержкой проверки пар компонентов (CPU + MB)
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
            print(f" Загружено {len(self.components_db)} компонентов")
            return True
        except Exception as e:
            self.components_db = []
            print(f" База компонентов не загружена: {e}")
            return False
    
    def load_model(self):
        try:
            model_path = os.path.join(self.base_dir, 'models', 'mlp_model.pt')
            if not os.path.exists(model_path):
                model_path = os.path.join(self.base_dir, 'src', 'mlp_model.pt')
            if not os.path.exists(model_path):
                print(" Модель не найдена")
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
                print(" scaler.pkl не найден")
                return False
            with open(scaler_path, 'rb') as f:
                saved = pickle.load(f)
            self.scaler = saved['scaler']
            self.label_encoders = saved['label_encoders']
            self.is_loaded = True
            print(" Модель и данные загружены успешно!")
            return True
        except Exception as e:
            print(f" Ошибка загрузки: {e}")
            return False
    
    def find_component_by_name(self, name):
        """Поиск компонента по названию (точное или частичное совпадение)"""
        name_lower = name.lower()
        for comp in self.components_db:
            comp_name = str(comp.get('Name', '')).lower()
            if name_lower in comp_name or comp_name in name_lower:
                return comp
        return None
    
    def smart_search(self, query):
        """Умный поиск с приоритетом точных совпадений"""
        results = []
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        is_cpu_query = any(word in ['ryzen', 'core', 'i3', 'i5', 'i7', 'i9', 'athlon', 'phenom', 'pentium', 'celeron', 'xeon'] for word in words)
        is_gpu_query = any(word in ['gtx', 'rtx', 'radeon', 'rx', 'geforce'] for word in words)
        is_ram_query = any(word in ['ddr', 'kingston', 'corsair', 'g.skill'] for word in words)
        is_socket_query = any(word in ['am4', 'am5', 'lga1700', 'lga1200', 'lga1151', 'lga1150', 'lga1155', 'lga1156', 'lga1366', 'socket', 'pcie', 'ddr2', 'ddr3', 'ddr4', 'ddr5'] for word in words)
        
        exact_cpu_model = None
        exact_cpu_series = None
        
        if is_cpu_query:
            ryzen_match = re.search(r'ryzen\s+([3579]|1[0-9])', query_lower)
            if ryzen_match:
                exact_cpu_model = f"ryzen {ryzen_match.group(1)}"
                exact_cpu_series = ryzen_match.group(1)
            if not exact_cpu_model:
                intel_match = re.search(r'(i[3579]|i[1-9][0-9])', query_lower)
                if intel_match:
                    exact_cpu_model = intel_match.group(0)
        
        exact_socket = None
        if is_socket_query:
            socket_patterns = ['am4', 'am5', 'lga1700', 'lga1200', 'lga1151', 'lga1150', 'lga1155', 'lga1156', 'lga1366', 'socket 775', 'pcie x16', 'ddr2', 'ddr3', 'ddr4', 'ddr5']
            for pattern in socket_patterns:
                if pattern in query_lower:
                    exact_socket = pattern
                    break
        
        for comp in self.components_db:
            name = str(comp.get('Name', '')).lower()
            comp_type = str(comp.get('Type', '')).lower()
            socket = str(comp.get('Socket', '')).lower()
            
            if name.startswith('#') or name.startswith('==='):
                continue
            
            if exact_socket:
                if exact_socket in socket or socket == exact_socket:
                    results.append(comp)
                continue
            
            if is_cpu_query and comp_type != 'cpu':
                continue
            if is_gpu_query and comp_type != 'gpu':
                continue
            if is_ram_query and comp_type != 'ram':
                continue
            
            if is_cpu_query and exact_cpu_model:
                if exact_cpu_model in name:
                    results.append(comp)
                    continue
                if 'ryzen' in query_lower and 'ryzen' in name and exact_cpu_series:
                    if re.search(f'ryzen\\s+{exact_cpu_series}', name):
                        results.append(comp)
                        continue
                continue
            
            if query_lower in name:
                results.append(comp)
                continue
            
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
        
        seen = set()
        unique_results = []
        for comp in results:
            comp_id = comp.get('Name', '')
            if comp_id not in seen:
                seen.add(comp_id)
                unique_results.append(comp)
        
        if is_cpu_query and 'ryzen' in query_lower:
            ryzen_series_match = re.search(r'ryzen\s+([3579]|1[0-9])', query_lower)
            if ryzen_series_match:
                target_series = ryzen_series_match.group(1)
                filtered = []
                for comp in unique_results:
                    name = str(comp.get('Name', '')).lower()
                    if 'ryzen' in name:
                        if re.search(f'ryzen\\s+{target_series}', name):
                            filtered.append(comp)
                        elif target_series in name and 'ryzen' in name:
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
    
    def predict_pair(self, cpu_name, mb_name):
        """Проверка совместимости пары CPU + Материнская плата"""
        # Ищем компоненты
        cpu = self.find_component_by_name(cpu_name)
        mb = self.find_component_by_name(mb_name)
        
        if not cpu:
            return f" Процессор '{cpu_name}' не найден"
        if not mb:
            return f" Материнская плата '{mb_name}' не найдена"
        
        # Проверяем типы
        if cpu.get('Type') != 'CPU':
            return f" '{cpu.get('Name')}' не является процессором"
        if mb.get('Type') != 'MB':
            return f" '{mb.get('Name')}' не является материнской платой"
        
        cpu_socket = cpu.get('Socket', '')
        mb_socket = mb.get('Socket', '')
        cpu_name_full = cpu.get('Name', '')
        mb_name_full = mb.get('Name', '')
        
        # Базовая проверка сокетов
        socket_match = cpu_socket == mb_socket
        
        # Получаем предсказания для каждого компонента
        cpu_pred, cpu_prob = self.predict_component(cpu)
        mb_pred, mb_prob = self.predict_component(mb)
        
        # Формируем результат
        result_text = ""
        result_text += f"\n╔══════════════════════════════════════════════════════════════╗"
        result_text += f"\n║              ПРОВЕРКА СВЯЗКИ КОМПОНЕНТОВ                     ║"
        result_text += f"\n╠══════════════════════════════════════════════════════════════╣"
        result_text += f"\n║  Процессор: {cpu_name_full[:40]}" + " "*(40-len(cpu_name_full[:40])) + "║"
        result_text += f"\n║  Сокет CPU: {cpu_socket}" + " "*(51-len(cpu_socket)) + "║"
        result_text += f"\n║  Материнская плата: {mb_name_full[:35]}" + " "*(45-len(mb_name_full[:35])) + "║"
        result_text += f"\n║  Сокет MB: {mb_socket}" + " "*(51-len(mb_socket)) + "║"
        result_text += f"\n╠══════════════════════════════════════════════════════════════╣"
        
        # Проверка сокетов
        if socket_match:
            result_text += f"\n║   Сокеты совпадают: {cpu_socket} == {mb_socket}" + " "*(30-len(cpu_socket)-len(mb_socket)) + "║"
        else:
            result_text += f"\n║   Сокеты НЕ совпадают: {cpu_socket} != {mb_socket}" + " "*(30-len(cpu_socket)-len(mb_socket)) + "║"
        
        # Индивидуальные предсказания
        if cpu_pred is not None:
            cpu_status = "" if cpu_pred == 1 else ""
            result_text += f"\n║  CPU совместимость: {cpu_status} {cpu_prob*100:.1f}%{' '*(40-len(str(int(cpu_prob*100))))}║"
        if mb_pred is not None:
            mb_status = "" if mb_pred == 1 else ""
            result_text += f"\n║  MB совместимость:  {mb_status} {mb_prob*100:.1f}%{' '*(40-len(str(int(mb_prob*100))))}║"
        
        # Общий вердикт
        result_text += f"\n╠══════════════════════════════════════════════════════════════╣"
        
        # Логика определения общей совместимости
        if socket_match and cpu_pred == 1 and mb_pred == 1:
            verdict = " СОВМЕСТИМЫ"
            confidence = (cpu_prob + mb_prob) / 2 * 100
        elif socket_match and (cpu_pred == 1 or mb_pred == 1):
            verdict = " ВОЗМОЖНО СОВМЕСТИМЫ (требуется проверка)"
            confidence = (cpu_prob + mb_prob) / 2 * 100
        else:
            verdict = " НЕ СОВМЕСТИМЫ"
            confidence = (cpu_prob + mb_prob) / 2 * 100
        
        result_text += f"\n║  Общий вердикт: {verdict}" + " "*(48-len(verdict)) + "║"
        result_text += f"\n║  Уверенность: {confidence:.1f}%{' '*(48-len(f'{confidence:.1f}%'))}║"
        result_text += f"\n╚══════════════════════════════════════════════════════════════╝"
        
        return result_text
    
    def predict_with_params(self, type_comp, socket, tdp, price):
        if not self.is_loaded:
            return "Модель не загружена"
        try:
            if type_comp not in self.label_encoders['Type'].classes_:
                return f" Тип '{type_comp}' не найден. Доступные типы: {self.label_encoders['Type'].classes_.tolist()}"
            if socket not in self.label_encoders['Socket'].classes_:
                return f" Сокет '{socket}' не найден. Доступные сокеты: {self.label_encoders['Socket'].classes_.tolist()}"
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
            return f" Ошибка: {e}"
    
    def show_full_group(self, comp_type):
        if not self.last_grouped:
            return " Сначала выполните поиск командой predict"
        comp_type_lower = comp_type.lower()
        found = False
        for key in self.last_grouped.keys():
            if key.lower() == comp_type_lower:
                comp_type = key
                found = True
                break
        if not found:
            available = ", ".join(self.last_grouped.keys())
            return f" Тип '{comp_type}' не найден. Доступны: {available}"
        comps = self.last_grouped[comp_type]
        output = f"\n {comp_type.upper()} (все {len(comps)} компонентов):\n" + "="*60 + "\n"
        for i, comp in enumerate(comps, 1):
            name = str(comp.get('Name', 'Unknown'))
            socket = str(comp.get('Socket', 'N/A'))
            tdp = comp.get('TDP', 0)
            price = comp.get('Price', 0)
            pred, prob = self.predict_component(comp)
            output += f"\n{i}. {name}\n"
            output += f"     Сокет: {socket}, TDP: {tdp}Вт, Цена: {price}руб\n"
            if pred is not None:
                result = " СОВМЕСТИМ" if pred == 1 else " НЕ СОВМЕСТИМ"
                output += f"     Совместимость: {result} (уверенность: {prob*100:.1f}%)\n"
        return output
    
    def show_search_results(self, query):
        results = self.smart_search(query)
        self.last_search_results = results
        self.last_search_query = query
        if not results:
            return f" Компоненты по запросу '{query}' не найдены"
        filtered_results = []
        for comp in results:
            name = str(comp.get('Name', ''))
            if name.startswith('#') or name.startswith('==='):
                continue
            if comp.get('TDP', '') == '' or comp.get('Price', '') == '':
                continue
            filtered_results.append(comp)
        if not filtered_results:
            return f" По запросу '{query}' найдены только служебные строки"
        output = f"\n Найдено {len(filtered_results)} компонентов по запросу '{query}':\n" + "="*60 + "\n"
        grouped = {}
        for comp in filtered_results:
            comp_type = str(comp.get('Type', 'Unknown'))
            if comp_type not in grouped:
                grouped[comp_type] = []
            grouped[comp_type].append(comp)
        self.last_grouped = grouped
        for comp_type in sorted(grouped.keys()):
            count = len(grouped[comp_type])
            output += f"\n {comp_type} ({count}):\n"
            for i, comp in enumerate(grouped[comp_type][:5], 1):
                name = str(comp.get('Name', 'Unknown'))
                socket = str(comp.get('Socket', 'N/A'))
                tdp = comp.get('TDP', 0)
                price = comp.get('Price', 0)
                pred, prob = self.predict_component(comp)
                output += f"\n  {i}. {name}\n"
                output += f"     Сокет: {socket}, TDP: {tdp}Вт, Цена: {price}руб\n"
                if pred is not None:
                    result = " СОВМЕСТИМ" if pred == 1 else " НЕ СОВМЕСТИМ"
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
║  ─────────────────────────────────────────────────────────── ║
║  predict <название_компонента>                               ║
║    - Поиск и проверка совместимости компонента               ║
║    - Пример: predict ryzen 5                                 ║
║    - Пример: predict am4                                     ║
║                                                              ║
║  pair cpu <название> mb <название>                           ║
║    - Проверка совместимости пары CPU + MB                    ║
║    - Пример: pair cpu "Ryzen 5 5600X" mb "B550"              ║
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
            return f" Доступные типы: {', '.join(self.label_encoders['Type'].classes_.tolist())}"
        return " Данные не загружены"
    
    def show_sockets(self):
        if self.label_encoders:
            return f" Доступные сокеты: {', '.join(self.label_encoders['Socket'].classes_.tolist())}"
        return " Данные не загружены"
    
    def run(self):
        print("=" * 60)
        print(" УМНЫЙ ЧАТ-БОТ ДЛЯ ПРОВЕРКИ СОВМЕСТИМОСТИ КОМПОНЕНТОВ")
        print("=" * 60)
        
        if not self.is_loaded:
            print(" Модель не загружена. Попробуйте перезапустить.")
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
                        print(" Используйте:")
                        print("  predict <название_компонента>")
                        print("  predict <тип> <сокет> <TDP> <цена>")
                        print("  predict am4")
                
                elif cmd in ["pair"]:
                    # Формат: pair cpu "название" mb "название"
                    # Ищем cpu и mb
                    try:
                        # Простой парсинг: ищем cpu и mb в аргументах
                        args_str = " ".join(args)
                        
                        # Ищем позиции cpu и mb
                        cpu_match = re.search(r'cpu\s+([^m]+?)(?:\s+mb|$)', args_str, re.IGNORECASE)
                        mb_match = re.search(r'mb\s+(.+)$', args_str, re.IGNORECASE)
                        
                        if cpu_match and mb_match:
                            cpu_name = cpu_match.group(1).strip()
                            mb_name = mb_match.group(1).strip()
                            
                            # Убираем лишние кавычки
                            cpu_name = cpu_name.strip('"').strip("'")
                            mb_name = mb_name.strip('"').strip("'")
                            
                            if cpu_name and mb_name:
                                result = self.predict_pair(cpu_name, mb_name)
                                print(result)
                            else:
                                print("Не удалось распознать названия компонентов")
                        else:
                            print(" Используйте: pair cpu <название> mb <название>")
                            print("Пример: pair cpu \"Ryzen 5 5600X\" mb \"B550\"")
                    except Exception as e:
                        print(f" Ошибка при разборе команды: {e}")
                        print("Пример: pair cpu \"Ryzen 5 5600X\" mb \"B550\"")
                
                elif cmd in ["more", "m"]:
                    if len(args) != 1:
                        print(" Используйте: more <тип>")
                        print("Пример: more CPU")
                        continue
                    result = self.show_full_group(args[0])
                    print(result)
                
                elif cmd in ["list_types", "types", "типы"]:
                    print(self.show_types())
                
                elif cmd in ["list_sockets", "sockets", "сокеты"]:
                    print(self.show_sockets())
                
                else:
                    print(f" Неизвестная команда: {cmd}")
                    print("Введите 'help' для списка команд")
                
            except KeyboardInterrupt:
                print("\n\nДо свидания!")
                break
            except Exception as e:
                print(f" Ошибка: {e}")

def main():
    bot = SmartCompatibilityBot()
    bot.run()

if __name__ == "__main__":
    main()