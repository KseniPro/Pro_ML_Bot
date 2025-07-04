import logging
import pandas as pd
import numpy as np
import html

from aiogram import F, types, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message, KeyboardButton, CallbackQuery
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from io import BytesIO
from aiogram.fsm.context import FSMContext
from handlers.states import DataStates, MLStates
from aiogram.types import BufferedInputFile
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, accuracy_score, 
                           classification_report, silhouette_score)
from io import BytesIO
from parse_params import *
from matplotlib import pyplot as plt
import seaborn as sns
import os
from aiogram.types import ReplyKeyboardMarkup

user_data = {}

MODEL_PARAMS = {
    'Linear Regression': 'fit_intercept=True',
    'Logistic Regression': 'C=1.0, max_iter=100',
    'Random Forest': 'n_estimators=100, max_depth=None',
    'XGBoost': 'n_estimators=100, max_depth=3, learning_rate=0.1',
    'LightGBM': 'n_estimators=100, max_depth=-1, learning_rate=0.1',
    'SVM': 'C=1.0, kernel=rbf',
    'K-Means': 'n_clusters=3, max_iter=300'
}

def create_main_keyboard():
    builder = ReplyKeyboardBuilder()
    builder.add(KeyboardButton(text='Показать первые N строк'))
    builder.add(KeyboardButton(text='Анализ данных'))
    builder.add(KeyboardButton(text='Предобработка данных'))
    builder.add(KeyboardButton(text='Выбрать ML модель'))
    builder.adjust(2, 2)
    return builder.as_markup(resize_keyboard=True)


async def create_ml_task_keyboard():
    print('create_ml_task_keyboard')
    builder = ReplyKeyboardBuilder()
    builder.add(KeyboardButton(text='Регрессия'))
    builder.add(KeyboardButton(text='Классификация'))
    builder.add(KeyboardButton(text='Кластеризация'))
    builder.add(KeyboardButton(text='Назад'))
    builder.adjust(2, 2)
    return builder.as_markup(resize_keyboard=True)

async def create_model_keyboard(task_type: str):
    print('create_model_keyboard')
    builder = ReplyKeyboardBuilder()
    
    if task_type == 'Регрессия':
        models = ['Linear Regression', 'Random Forest', 'XGBoost', 'Назад']
    elif task_type == 'Классификация':
        models = ['Logistic Regression', 'Random Forest', 'SVM', 'Назад']
    elif task_type == 'Кластеризация':
        models = ['K-Means', 'Назад']
    
    for model in models:
        builder.add(KeyboardButton(text=model))
    
    builder.adjust(2, 2)
    return builder.as_markup(resize_keyboard=True)

async def create_results_excel(user_id: int):
    """Генерирует Excel файл с результатами"""
    data = user_data[user_id]
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Лист с предсказаниями
        data['predictions'].to_excel(writer, sheet_name='Predictions', index=False)
        
        # Лист с метриками
        metrics_df = pd.DataFrame(data['metrics'].items(), columns=['Metric', 'Value'])
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    
    output.seek(0)
    return output

async def register_data_handlers(dp: Dispatcher):
    @dp.message(Command('start'))
    async def cmd_start(message: Message):
        try: 
            user_name = message.from_user.first_name
            await message.answer(
                f'Добрый день, <b>{user_name}</b>! 👋\n\n'
                'Это Telegram-бот для автоматизации ML-процессов.\n'
                'Пожалуйста, загрузите файл в формате CSV или Excel.',
                parse_mode="HTML")
        except Exception as e:
            logging.error(f"Ошибка при старте: {e}")
            await message.answer("⚠️ Произошла ошибка. Попробуйте позже.")


    @dp.message(F.document)
    async def handle_document(message: Message):
        try:
            file_id = message.document.file_id
            file = await message.bot.get_file(file_id)
            file_path = file.file_path
            
            if not (file_path.endswith('.csv') or file_path.endswith('.xlsx')):
                await message.answer('🚫 Пожалуйста, загрузите файл в формате CSV или Excel.')
                return
            
            file_bytes = BytesIO()
            await message.bot.download_file(file_path, file_bytes)
            file_bytes.seek(0)
            
            df = pd.read_csv(file_bytes) if file_path.endswith('.csv') else pd.read_excel(file_bytes)
            user_data[message.from_user.id] = {'df': df}
            
            await message.answer(
                '✅ Файл успешно загружен! Выберите действие:',
                reply_markup=create_main_keyboard())
                # reply_markup=builder.as_markup(resize_keyboard=True))
                
        except Exception as e:
            logging.error(f'Ошибка обработки файла: {e}')
            await message.answer('⚠️ Произошла ошибка при обработке файла.')

    
    @dp.message(F.text == 'Анализ данных')
    async def show_data_analysis(message: Message):
        if message.from_user.id not in user_data:
            await message.answer('ℹ️ Сначала загрузите файл.')
            return
        df = user_data[message.from_user.id]['df']
        info = ('📊 <b>Анализ данных:</b>\n\n'
            f'• Строк: {df.shape[0]}\n'
            f'• Столбцов: {df.shape[1]}\n\n'
            '<b>Пропущенные значения:</b>\n'
            f'<pre>{df.isna().sum().to_string()}</pre>\n\n'
            '<b>Типы данных:</b>\n'
            f'<pre>{df.dtypes.to_string()}</pre>')
        await message.answer(info, parse_mode='HTML')
        # Визуализация: только по числовым столбцам
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        user_id = message.from_user.id
        try:
            num_df = df.select_dtypes(include=['number'])
            if num_df.shape[1] == 0:
                await message.answer('Нет числовых столбцов для визуализации.')
                return
            plt.figure(figsize=(8, 6))
            sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm')
            plt.title('Корреляционная матрица')
            corr_path = f'logs/corr_{user_id}.png'
            plt.tight_layout()
            plt.savefig(corr_path)
            plt.close()
            with open(corr_path, 'rb') as img_file:
                photo = BufferedInputFile(img_file.read(), filename=os.path.basename(corr_path))
                await message.answer_photo(photo, caption='Корреляционная матрица')
            os.remove(corr_path)
            # Pairplot (если не слишком много признаков)
            if num_df.shape[1] <= 8 and num_df.shape[1] > 1:
                pairplot_path = f'logs/pairplot_{user_id}.png'
                sns.pairplot(num_df)
                plt.savefig(pairplot_path)
                plt.close()
                with open(pairplot_path, 'rb') as img_file:
                    photo = BufferedInputFile(img_file.read(), filename=os.path.basename(pairplot_path))
                    await message.answer_photo(photo, caption='Pairplot')
                os.remove(pairplot_path)
        except Exception as e:
            await message.answer(f'Ошибка при построении графиков: {e}')

    @dp.message(F.text == 'Показать первые N строк')
    async def ask_for_rows_number(message: Message, state: FSMContext):
        if message.from_user.id not in user_data:
            await message.answer('ℹ️ Сначала загрузите файл.')
            return
        
        await message.answer('📝 Введите количество строк для просмотра (от 1 до 50):',
            reply_markup=types.ReplyKeyboardRemove())
        await state.set_state(DataStates.waiting_for_rows)

    @dp.message(DataStates.waiting_for_rows)
    async def show_custom_rows(message: Message, state: FSMContext):
        try:
            n = int(message.text)
            if n < 1 or n > 50:
                raise ValueError
            
            df = user_data[message.from_user.id]['df']
            table = []
            headers = [f"<b>{col}</b>" for col in df.columns]
            table.append("|".join(headers))
            
            for _, row in df.head(n).iterrows():
                table.append("|".join(str(x) for x in row.values))
            
            response = (f"<b>Первые {n} строк данных:</b>\n"
                f"<code>\n" + "\n".join(table) + "\n</code>")
            
            await message.answer(response, parse_mode='HTML')
            await message.answer('Выберите следующее действие:',
                reply_markup=create_main_keyboard())
            await state.clear()
                
        except ValueError:
            await message.answer('⚠️ Пожалуйста, введите число от 1 до 50.')
            

    # Обработчик для всех остальных текстовых сообщений
    # @dp.message(F.text)
    # async def handle_other_text(message: Message):
    #     if message.from_user.id in user_data:
    #         builder = ReplyKeyboardBuilder()
    #         builder.add(KeyboardButton(text='Показать первые N строк'))
    #         builder.add(KeyboardButton(text='Информация о данных'))
            
    #         await message.answer('ℹ️ Пожалуйста, используйте кнопки для работы с данными.',
    #             reply_markup=builder.as_markup(resize_keyboard=True))
    #     else:
    #         await message.answer('🚫 Пожалуйста, сначала загрузите файл в формате CSV или Excel.')
            
            

    @dp.message(F.text == 'Предобработка данных')
    async def preprocess_menu(message: Message, state: FSMContext):
        if message.from_user.id not in user_data:
            await message.answer('ℹ️ Сначала загрузите файл.')
            return

        builder = ReplyKeyboardBuilder()
        builder.add(KeyboardButton(text='Удалить дубликаты'))
        builder.add(KeyboardButton(text='Обработка пропусков'))
        builder.add(KeyboardButton(text='Назад'))
        await message.answer('Выберите тип предобработки.',
            reply_markup=builder.as_markup(resize_keyboard=True))
        await state.set_state(DataStates.waiting_preprocess_action)

    # === Удаление дубликатов ===
    @dp.message(F.text == 'Удалить дубликаты', DataStates.waiting_preprocess_action)
    async def duplicates_menu(message: Message, state: FSMContext):
        builder = InlineKeyboardBuilder()
        builder.button(text="По всей строке", callback_data="dup_full")
        builder.button(text="Выбрать столбцы", callback_data="dup_select")
        await message.answer("Как удалить дубликаты?",
            reply_markup=builder.as_markup())
        await state.set_state(DataStates.waiting_dup_action)

    @dp.callback_query(F.data.startswith('dup_'), DataStates.waiting_dup_action)
    async def process_duplicates(callback: CallbackQuery, state: FSMContext):
        action = callback.data.split('_')[1]
        df = user_data[callback.from_user.id]['df'].copy()
        
        if action == "full":
            initial_count = len(df)
            df = df.drop_duplicates()
            removed = initial_count - len(df)
            
            await callback.message.answer(f"✅ Удалено {removed} полных дубликатов.\n"
                f"Осталось строк: {len(df)}.")
        elif action == "select":
            columns = df.columns.tolist()
            builder = InlineKeyboardBuilder()
            
            for col in columns:
                builder.button(text=f"✓ {col}" if col in (await state.get_data()).get('selected_columns', []) else col,
                    callback_data=f"col_{col}")
            
            builder.button(text="✅ Применить", callback_data="dup_confirm")
            builder.adjust(2, repeat=True)
            
            await callback.message.edit_text("Выберите столбцы для проверки дубликатов:",
                reply_markup=builder.as_markup())
            await state.update_data(selected_columns=[])
            await state.set_state(DataStates.waiting_columns_selection)
            return
        
        user_data[callback.from_user.id]['df'] = df
        await state.clear()
        await callback.message.answer('Выберите следующее действие:',
                reply_markup=create_main_keyboard())
        await callback.answer()

    # Обработка выбора столбцов
    @dp.callback_query(F.data.startswith('col_'), DataStates.waiting_columns_selection)
    async def select_columns(callback: CallbackQuery, state: FSMContext):
        column = callback.data.split('_')[1]
        data = await state.get_data()
        selected = data.get('selected_columns', [])
        
        if column in selected:
            selected.remove(column)
        else:
            selected.append(column)
        
        await state.update_data(selected_columns=selected)
        
        # Обновляем клавиатуру
        df = user_data[callback.from_user.id]['df']
        builder = InlineKeyboardBuilder()
        
        for col in df.columns:
            prefix = "✓ " if col in selected else ""
            builder.button(text=f"{prefix}{col}", callback_data=f"col_{col}")
        
        builder.button(text="✅ Применить", callback_data="dup_confirm")
        builder.adjust(2, repeat=True)
        
        await callback.message.edit_reply_markup(reply_markup=builder.as_markup())
        await callback.answer()

    # Применение выбора столбцов
    @dp.callback_query(F.data == 'dup_confirm', DataStates.waiting_columns_selection)
    async def apply_duplicates_removal(callback: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        selected = data.get('selected_columns', [])
        
        if not selected:
            await callback.answer("❌ Не выбрано ни одного столбца.", show_alert=True)
            return
        
        df = user_data[callback.from_user.id]['df'].copy()
        initial_count = len(df)
        df = df.drop_duplicates(subset=selected)
        removed = initial_count - len(df)
        
        user_data[callback.from_user.id]['df'] = df
        await callback.message.answer(f"✅ Удалено {removed} дубликатов по столбцам: {', '.join(selected)}.\n"
            f"Осталось строк: {len(df)}")
        await state.clear()
        await callback.message.answer('Выберите следующее действие:',
                reply_markup=create_main_keyboard())
        await callback.answer()

    # === Обработка пропусков ===
    @dp.message(F.text == 'Обработка пропусков', DataStates.waiting_preprocess_action)
    async def handle_missing_data(message: Message, state: FSMContext):
        builder = InlineKeyboardBuilder()
        builder.button(text="Удалить строки с пропусками", callback_data="missing_drop")
        builder.button(text="Заполнить пропуски", callback_data="missing_fill")
        await message.answer("Как обработать пропущенные значения?",
            reply_markup=builder.as_markup())
        await state.set_state(DataStates.waiting_missing_action)

    @dp.callback_query(F.data.startswith('missing_'), DataStates.waiting_missing_action)
    async def process_missing_action(callback: CallbackQuery, state: FSMContext):
        action = callback.data.split('_')[1]
        df = user_data[callback.from_user.id]['df'].copy()
        
        if action == "drop":
            initial_count = len(df)
            df = df.dropna()
            removed = initial_count - len(df)
            
            await callback.message.answer(
                f"✅ Удалено {removed} строк с пропусками.\n"
                f"Осталось строк: {len(df)}.")
        elif action == "fill":
            builder = InlineKeyboardBuilder()
            builder.button(text="Среднее значение", callback_data="fill_mean")
            builder.button(text="Медиана", callback_data="fill_median")
            builder.button(text="Мода", callback_data="fill_mode")
            builder.button(text="Константа", callback_data="fill_const")
            
            await callback.message.edit_text(
                "Выберите метод заполнения пропусков:",
                reply_markup=builder.as_markup())
            await state.set_state(DataStates.waiting_fill_method)
            return
        
        user_data[callback.from_user.id]['df'] = df
        await state.clear()
        await callback.message.answer('Выберите следующее действие:',
                reply_markup=create_main_keyboard())
        await callback.answer()

    @dp.callback_query(F.data.startswith('fill_'), DataStates.waiting_fill_method)
    async def process_fill_method(callback: CallbackQuery, state: FSMContext):
        method = callback.data.split('_')[1]
        user_id = callback.from_user.id
        df = user_data[user_id]['df']
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if method == "const":
            await callback.message.answer("Введите значение для заполнения пропусков:")
            await state.update_data(fill_method=method)
            return
        
        numeric_data = df[numeric_cols].copy()
        
        if method == "mean":
            fill_values = numeric_data.mean()
        elif method == "median":
            fill_values = numeric_data.median()
        elif method == "mode":
            fill_values = numeric_data.mode().iloc[0]
        
        df[numeric_cols] = numeric_data.fillna(fill_values)
        user_data[user_id]['df'] = df
        await callback.message.answer(f"✅ Пропуски заполнены ({method}.)")
        await state.clear()
        await callback.message.answer('Выберите следующее действие:',
                reply_markup=create_main_keyboard())
        await callback.answer()


    @dp.message(DataStates.waiting_fill_method)
    async def process_fill_const(message: Message, state: FSMContext):
        try:
            value = float(message.text)
            user_id = message.from_user.id
            df = user_data[user_id]['df']
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(value)
            
            await message.answer(f"✅ Пропуски заполнены значением {value}.")
            await state.clear()
            await message.answer('Выберите следующее действие:',
                reply_markup=create_main_keyboard())
            
        except ValueError:
            await message.answer("❌ Введите числовое значение.")


    @dp.message(F.text == 'Назад', DataStates.waiting_preprocess_action)
    async def back_from_preprocess(message: Message, state: FSMContext):
        await message.answer('Главное меню',
            reply_markup=create_main_keyboard())
        await state.clear()




# new
    @dp.message(F.text == 'Выбрать ML модель')
    async def select_ml_task(message: Message, state: FSMContext):
        if message.from_user.id not in user_data:
            await message.answer('ℹ️ Сначала загрузите файл.')
            return
            
        await message.answer(
            'Выберите тип задачи машинного обучения:',
            reply_markup=await create_ml_task_keyboard())
        await state.set_state(MLStates.selecting_ml_task)

    @dp.message(MLStates.selecting_ml_task, F.text.in_(['Регрессия', 'Классификация', 'Кластеризация']))
    async def select_model_type(message: Message, state: FSMContext):
        task_type = message.text
        user_id = message.from_user.id
        
        if user_id not in user_data:
            user_data[user_id] = {}
        user_data[user_id]['ml_task'] = task_type
        
        await message.answer(
            f'Вы выбрали {task_type}. Теперь выберите модель:',
            reply_markup=await create_model_keyboard(task_type))
        await state.set_state(MLStates.selecting_model)


    @dp.message(MLStates.selecting_model, F.text.in_(MODEL_PARAMS.keys()))
    async def configure_model(message: Message, state: FSMContext):
        user_id = message.from_user.id
        model_name = message.text
        user_data[user_id]['ml_model'] = model_name
        
        # Добавляем кнопку 'По умолчанию'
        builder = ReplyKeyboardBuilder()
        builder.add(KeyboardButton(text='По умолчанию'))
        await message.answer(
            f'Настройте параметры для {model_name} (через запятую):\n'
            f'Пример: {MODEL_PARAMS[model_name]}\n'
            'Или выберите "По умолчанию" кнопкой ниже',
            reply_markup=builder.as_markup(resize_keyboard=True)
        )
        await state.set_state(MLStates.configuring_model)

    @dp.message(MLStates.configuring_model, F.text.casefold() == 'по умолчанию')
    async def use_default_params(message: Message, state: FSMContext):
        user_id = message.from_user.id
        model_name = user_data[user_id]['ml_model']
        user_data[user_id]['model_params'] = MODEL_PARAMS[model_name]
        
        # Новый этап: выбор признаков
        df = user_data[user_id]['df']
        columns = df.columns
        dtypes = df.dtypes
        col_list = [f"{i+1}. {col}: {dtypes[col]}" for i, col in enumerate(columns)]
        await message.answer(
            'Модель будет использована с параметрами по умолчанию:\n'
            f'{MODEL_PARAMS[model_name]}\n'
            'Теперь выберите номера признаков (через запятую):\n'
            + '\n'.join(col_list)
        )
        await state.set_state(MLStates.selecting_features)

    @dp.message(MLStates.configuring_model)
    async def process_custom_params(message: Message, state: FSMContext):
        user_id = message.from_user.id
        model_name = user_data[user_id]['ml_model']
        
        if message.text.lower() in ['назад', 'back']:
            await message.answer('Выберите модель:',
                reply_markup=await create_model_keyboard(user_data[user_id]['ml_task']))
            await state.set_state(MLStates.selecting_model)
            return
            
        try:
            params = {}
            for param in message.text.split(','):
                key, value = param.strip().split('=')
                params[key.strip()] = value.strip()
            
            user_data[user_id]['model_params'] = params
            
            # Новый этап: выбор признаков
            df = user_data[user_id]['df']
            columns = df.columns
            dtypes = df.dtypes
            col_list = [f"{i+1}. {col}: {dtypes[col]}" for i, col in enumerate(columns)]
            await message.answer(
                f'Параметры для {model_name} установлены:\n'
                f'{params}\n'
                'Теперь выберите номера признаков (через запятую):\n'
                + '\n'.join(col_list)
            )
            await state.set_state(MLStates.selecting_features)
            
        except Exception as e:
            await message.answer(
                f'⚠️ Ошибка: {str(e)}\n'
                f'Пример формата: n_estimators=100, max_depth=5\n'
                'Попробуйте снова или отправьте "по умолчанию"'
            )

    # Новый этап: выбор признаков
    @dp.message(MLStates.selecting_features)
    async def select_features(message: Message, state: FSMContext):
        user_id = message.from_user.id
        df = user_data[user_id]['df']
        columns = df.columns
        task_type = user_data[user_id].get('ml_task', '').lower()
        
        try:
            feature_idxs = [int(idx.strip())-1 for idx in message.text.split(',') if idx.strip().isdigit()]
            if not feature_idxs or any(idx < 0 or idx >= len(columns) for idx in feature_idxs):
                raise ValueError
            user_data[user_id]['feature_idxs'] = feature_idxs
            
            # Для кластеризации не нужен target
            if 'кластеризация' in task_type:
                # После выбора признаков для кластеризации сразу предлагаем предобработку
                builder = ReplyKeyboardBuilder()
                builder.add(KeyboardButton(text='Удалить дубликаты'))
                builder.add(KeyboardButton(text='Обработка пропусков'))
                builder.add(KeyboardButton(text='Продолжить обучение'))
                builder.add(KeyboardButton(text='Вернуться в меню'))
                builder.adjust(2, 2)
                await message.answer(
                    'Для кластеризации target не нужен. Перед обучением вы можете выполнить предобработку данных.\n'
                    'Выберите действие:',
                    reply_markup=builder.as_markup(resize_keyboard=True)
                )
                await state.set_state(MLStates.preprocessing_before_train)
            else:
                # Для регрессии и классификации просим выбрать target
                dtypes = df.dtypes
                col_list = [f"{i+1}. {col}: {dtypes[col]}" for i, col in enumerate(columns)]
                await message.answer(
                    'Теперь выберите номер таргета (целевой переменной):\n'
                    + '\n'.join(col_list)
                )
                await state.set_state(MLStates.selecting_target)
        except Exception:
            await message.answer('❌ Введите номера признаков через запятую, например: 1,2,3')

    # Новый этап: выбор таргета
    @dp.message(MLStates.selecting_target)
    async def select_target(message: Message, state: FSMContext):
        user_id = message.from_user.id
        df = user_data[user_id]['df']
        columns = df.columns
        try:
            target_idx = int(message.text.strip())-1
            if target_idx < 0 or target_idx >= len(columns):
                raise ValueError
            user_data[user_id]['target_idx'] = target_idx
            # После выбора таргета предлагаем предобработку
            builder = ReplyKeyboardBuilder()
            builder.add(KeyboardButton(text='Удалить дубликаты'))
            builder.add(KeyboardButton(text='Обработка пропусков'))
            builder.add(KeyboardButton(text='Продолжить обучение'))
            builder.add(KeyboardButton(text='Вернуться в меню'))
            builder.adjust(2, 2)
            await message.answer(
                'Перед обучением вы можете выполнить предобработку данных.\n'
                'Выберите действие:',
                reply_markup=builder.as_markup(resize_keyboard=True)
            )
            await state.set_state(MLStates.preprocessing_before_train)
        except Exception:
            await message.answer('❌ Введите номер столбца для таргета, например: 4')

    # Меню предобработки перед обучением
    @dp.message(MLStates.preprocessing_before_train)
    async def preprocessing_before_train(message: Message, state: FSMContext):
        user_id = message.from_user.id
        text = message.text.strip().lower()
        df = user_data[user_id]['df']
        if text == 'удалить дубликаты':
            builder = InlineKeyboardBuilder()
            builder.button(text="По всей строке", callback_data="dup_full_train")
            builder.button(text="Выбрать столбцы", callback_data="dup_select_train")
            await message.answer("Как удалить дубликаты?",
                reply_markup=builder.as_markup())
            await state.set_state(DataStates.waiting_dup_action)
        elif text == 'обработка пропусков':
            builder = InlineKeyboardBuilder()
            builder.button(text="Удалить строки с пропусками", callback_data="missing_drop_train")
            builder.button(text="Заполнить средним", callback_data="fill_mean_train")
            builder.button(text="Заполнить медианой", callback_data="fill_median_train")
            builder.button(text="Заполнить модой", callback_data="fill_mode_train")
            builder.button(text="Заполнить константой", callback_data="fill_const_train")
            await message.answer("Как обработать пропуски?",
                reply_markup=builder.as_markup())
            await state.set_state(DataStates.waiting_missing_action)
        elif text == 'продолжить обучение':
            await message.answer('Начинаю обучение...')
            await state.set_state(MLStates.training_model)
            await handle_model_training(message, state)
        elif text == 'вернуться в меню':
            await message.answer('Главное меню', reply_markup=create_main_keyboard())
            await state.clear()
        else:
            await message.answer('Пожалуйста, выберите действие из меню.')

    # Обработка дубликатов перед обучением
    @dp.callback_query(F.data.in_(["dup_full_train", "dup_select_train"]), DataStates.waiting_dup_action)
    async def process_duplicates_train(callback: CallbackQuery, state: FSMContext):
        user_id = callback.from_user.id
        df = user_data[user_id]['df'].copy()
        action = callback.data
        if action == "dup_full_train":
            initial_count = len(df)
            df = df.drop_duplicates()
            removed = initial_count - len(df)
            user_data[user_id]['df'] = df
            await callback.message.answer(f"✅ Удалено {removed} полных дубликатов. Осталось строк: {len(df)}.")
            await callback.message.answer('Выберите следующее действие:',
                reply_markup=ReplyKeyboardBuilder()
                    .add(KeyboardButton(text='Удалить дубликаты'))
                    .add(KeyboardButton(text='Обработка пропусков'))
                    .add(KeyboardButton(text='Продолжить обучение'))
                    .add(KeyboardButton(text='Вернуться в меню'))
                    .adjust(2, 2)
                    .as_markup(resize_keyboard=True))
            await state.set_state(MLStates.preprocessing_before_train)
            await callback.answer()
        elif action == "dup_select_train":
            columns = df.columns.tolist()
            builder = InlineKeyboardBuilder()
            for col in columns:
                builder.button(text=col, callback_data=f"col_train_{col}")
            builder.button(text="✅ Применить", callback_data="dup_confirm_train")
            builder.adjust(2, repeat=True)
            await callback.message.answer("Выберите столбцы для проверки дубликатов:",
                reply_markup=builder.as_markup())
            await state.update_data(selected_columns=[])
            await state.set_state(DataStates.waiting_columns_selection)
            await callback.answer()

    # Выбор столбцов для удаления дубликатов перед обучением
    @dp.callback_query(F.data.startswith('col_train_'), DataStates.waiting_columns_selection)
    async def select_columns_train(callback: CallbackQuery, state: FSMContext):
        column = callback.data[len('col_train_'):]
        data = await state.get_data()
        selected = data.get('selected_columns', [])
        if column in selected:
            selected.remove(column)
        else:
            selected.append(column)
        await state.update_data(selected_columns=selected)
        df = user_data[callback.from_user.id]['df']
        builder = InlineKeyboardBuilder()
        for col in df.columns:
            prefix = "✓ " if col in selected else ""
            builder.button(text=f"{prefix}{col}", callback_data=f"col_train_{col}")
        builder.button(text="✅ Применить", callback_data="dup_confirm_train")
        builder.adjust(2, repeat=True)
        await callback.message.edit_reply_markup(reply_markup=builder.as_markup())
        await callback.answer()

    # Применение удаления дубликатов по выбранным столбцам перед обучением
    @dp.callback_query(F.data == 'dup_confirm_train', DataStates.waiting_columns_selection)
    async def apply_duplicates_removal_train(callback: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        selected = data.get('selected_columns', [])
        if not selected:
            await callback.answer("❌ Не выбрано ни одного столбца.", show_alert=True)
            return
        df = user_data[callback.from_user.id]['df'].copy()
        initial_count = len(df)
        df = df.drop_duplicates(subset=selected)
        removed = initial_count - len(df)
        user_data[callback.from_user.id]['df'] = df
        await callback.message.answer(f"✅ Удалено {removed} дубликатов по столбцам: {', '.join(selected)}. Осталось строк: {len(df)}")
        await callback.message.answer('Выберите следующее действие:',
            reply_markup=ReplyKeyboardBuilder()
                .add(KeyboardButton(text='Удалить дубликаты'))
                .add(KeyboardButton(text='Обработка пропусков'))
                .add(KeyboardButton(text='Продолжить обучение'))
                .add(KeyboardButton(text='Вернуться в меню'))
                .adjust(2, 2)
                .as_markup(resize_keyboard=True))
        await state.set_state(MLStates.preprocessing_before_train)
        await callback.answer()

    # Обработка пропусков перед обучением
    @dp.callback_query(F.data.in_(["missing_drop_train", "fill_mean_train", "fill_median_train", "fill_mode_train", "fill_const_train"]), DataStates.waiting_missing_action)
    async def process_missing_action_train(callback: CallbackQuery, state: FSMContext):
        user_id = callback.from_user.id
        df = user_data[user_id]['df']
        action = callback.data
        numeric_cols = df.select_dtypes(include=['number']).columns
        if action == "missing_drop_train":
            initial_count = len(df)
            df = df.dropna()
            removed = initial_count - len(df)
            user_data[user_id]['df'] = df
            await callback.message.answer(f"✅ Удалено {removed} строк с пропусками. Осталось строк: {len(df)}.")
        elif action == "fill_mean_train":
            fill_values = df[numeric_cols].mean()
            df[numeric_cols] = df[numeric_cols].fillna(fill_values)
            user_data[user_id]['df'] = df
            await callback.message.answer(f"✅ Пропуски заполнены средним.")
        elif action == "fill_median_train":
            fill_values = df[numeric_cols].median()
            df[numeric_cols] = df[numeric_cols].fillna(fill_values)
            user_data[user_id]['df'] = df
            await callback.message.answer(f"✅ Пропуски заполнены медианой.")
        elif action == "fill_mode_train":
            fill_values = df[numeric_cols].mode().iloc[0]
            df[numeric_cols] = df[numeric_cols].fillna(fill_values)
            user_data[user_id]['df'] = df
            await callback.message.answer(f"✅ Пропуски заполнены модой.")
        elif action == "fill_const_train":
            await callback.message.answer("Введите значение для заполнения пропусков:")
            await state.update_data(fill_method='const_train')
            return
        await callback.message.answer('Выберите следующее действие:',
            reply_markup=ReplyKeyboardBuilder()
                .add(KeyboardButton(text='Удалить дубликаты'))
                .add(KeyboardButton(text='Обработка пропусков'))
                .add(KeyboardButton(text='Продолжить обучение'))
                .add(KeyboardButton(text='Вернуться в меню'))
                .adjust(2, 2)
                .as_markup(resize_keyboard=True))
        await state.set_state(MLStates.preprocessing_before_train)
        await callback.answer()

    # Ввод значения для заполнения пропусков константой перед обучением
    @dp.message(DataStates.waiting_fill_method)
    async def process_fill_const_train(message: Message, state: FSMContext):
        data = await state.get_data()
        if data.get('fill_method') == 'const_train':
            try:
                value = float(message.text)
                user_id = message.from_user.id
                df = user_data[user_id]['df']
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(value)
                user_data[user_id]['df'] = df
                await message.answer(f"✅ Пропуски заполнены значением {value}.")
                await message.answer('Выберите следующее действие:',
                    reply_markup=ReplyKeyboardBuilder()
                        .add(KeyboardButton(text='Удалить дубликаты'))
                        .add(KeyboardButton(text='Обработка пропусков'))
                        .add(KeyboardButton(text='Продолжить обучение'))
                        .add(KeyboardButton(text='Вернуться в меню'))
                        .adjust(2, 2)
                        .as_markup(resize_keyboard=True))
                await state.set_state(MLStates.preprocessing_before_train)
            except ValueError:
                await message.answer("❌ Введите числовое значение.")
        else:
            # старый обработчик для других случаев
            pass

    @dp.message(MLStates.training_model)
    async def handle_model_training(message: Message, state: FSMContext):
        if message.text and message.text.strip().lower() == 'в меню':
            await message.answer('Главное меню', reply_markup=create_main_keyboard())
            await state.clear()
            return
        user_id = message.from_user.id
        try:
            data = user_data[user_id]
            df = data['df'].copy()
            model_name = data['ml_model']
            params = parse_params(data['model_params']) if isinstance(data['model_params'], str) else data['model_params']
            task_type = data['ml_task'].lower()
            feature_idxs = data.get('feature_idxs')
            target_idx = data.get('target_idx')
            
            # Обработка в зависимости от типа задачи
            if 'кластеризация' in task_type:
                # Для кластеризации используем только признаки, без target
                if feature_idxs is not None:
                    if not isinstance(feature_idxs, list):
                        feature_idxs = [feature_idxs]
                    X = df.iloc[:, feature_idxs]
                else:
                    # Если признаки не выбраны, используем все столбцы кроме последнего
                    X = df.iloc[:, :-1]
                
                if isinstance(X, pd.Series):
                    X = X.to_frame()
                
                # Для кластеризации не делаем train_test_split
                model = create_model_instance(model_name, params, task_type)
                model.fit(X)
                y_pred = model.predict(X)
                
                metrics = calculate_metrics(
                    y_true=None,  # Для кластеризации не нужен y_true
                    y_pred=y_pred,
                    task_type=task_type,
                    X=X
                )
                
                # Создаем DataFrame с результатами кластеризации
                results_df = pd.DataFrame({
                    'Cluster': y_pred
                })
                if hasattr(X, 'columns') and len(X.columns) <= 5:
                    for i, col in enumerate(X.columns):
                        results_df[f'Feature_{i+1}'] = X[col].values
                
            else:
                # Для регрессии и классификации используем и признаки, и target
                if feature_idxs is not None and target_idx is not None:
                    if not isinstance(feature_idxs, list):
                        feature_idxs = [feature_idxs]
                    all_idxs = feature_idxs + [target_idx]
                    temp_df = df.iloc[:, all_idxs]
                    X = temp_df.iloc[:, :-1]
                    y = temp_df.iloc[:, -1]
                else:
                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1]
                
                if isinstance(X, pd.Series):
                    X = X.to_frame()
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                if isinstance(X_train, pd.Series):
                    X_train = X_train.to_frame()
                if isinstance(X_test, pd.Series):
                    X_test = X_test.to_frame()
                
                model = create_model_instance(model_name, params, task_type)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                metrics = calculate_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    task_type=task_type
                )
                
                results_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred
                })
                if hasattr(X_test, 'columns') and len(X_test.columns) <= 5:
                    for i, col in enumerate(X_test.columns):
                        results_df[f'Feature_{i+1}'] = X_test[col].values
            
            user_data[user_id].update({
                'trained_model': model,
                'metrics': metrics,
                'predictions': results_df
            })
            
            metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            await message.answer(f"📊 Результаты обучения:\n{metrics_text}")
            excel_file = await create_results_excel(user_id)
            await message.answer_document(
                types.BufferedInputFile(
                    excel_file.getvalue(),
                    filename="model_results.xlsx"),
                caption="Результаты предсказаний")
            await state.clear()
        except Exception as e:
            err_text = html.escape(str(e))
            await message.answer(f"❌ Ошибка при обучении модели:\n<code>{err_text}</code>", parse_mode='HTML')
            await message.answer(
                "Выберите следующее действие:",
                reply_markup=create_main_keyboard())
            await state.clear()

    # Кнопка 'В меню' для возврата в главное меню из этапа обучения
    @dp.message(MLStates.training_model, F.text.casefold() == 'в меню')
    async def back_to_menu_from_training(message: Message, state: FSMContext):
        await message.answer('Главное меню', reply_markup=create_main_keyboard())
        await state.clear()

    @dp.message(F.text)
    async def handle_other_text(message: Message):
        if (message.from_user.id in user_data and 
            message.text not in ['Показать первые N строк', 
                            'Анализ данных',
                            'Предобработка данных',
                            'Выбрать ML модель']):

            await message.answer('ℹ️ Пожалуйста, используйте кнопки для работы с данными.',
                reply_markup=create_main_keyboard())
        elif message.from_user.id not in user_data:
            await message.answer('🚫 Пожалуйста, сначала загрузите файл в формате CSV или Excel.')  
            
    
      
    
    
    
    
