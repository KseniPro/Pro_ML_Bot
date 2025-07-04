from aiogram.fsm.state import State, StatesGroup

class DataStates(StatesGroup):
    waiting_for_rows = State()
    waiting_preprocess_action = State()  # Добавляем недостающее состояние
    waiting_dup_action = State()
    waiting_columns_selection = State()
    waiting_missing_action = State()
    waiting_fill_method = State()
    
class MLStates(StatesGroup):
    selecting_ml_task = State()
    selecting_model = State()
    configuring_model = State()
    training_model = State()
    analyzing_data = State()
    selecting_features = State()
    selecting_target = State()
    preprocessing_before_train = State()