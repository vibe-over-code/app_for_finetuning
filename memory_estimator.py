"""
Модуль для оценки требований к памяти перед обучением
"""
import torch
from transformers import AutoConfig
from huggingface_hub import model_info
import json

def estimate_model_memory(model_name, quantization_bits=4):
    """
    Оценивает память, необходимую для загрузки модели
    
    Args:
        model_name: Имя модели (путь или HuggingFace ID)
        quantization_bits: Битность квантования (4, 8 или None для без квантования)
    
    Returns:
        dict: Оценка памяти в GB
    """
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Получаем количество параметров
        if hasattr(config, 'num_parameters'):
            num_params = config.num_parameters()
        else:
            # Оценка на основе архитектуры
            if hasattr(config, 'hidden_size') and hasattr(config, 'num_hidden_layers'):
                # Примерная формула для transformer моделей
                hidden_size = config.hidden_size
                num_layers = config.num_hidden_layers
                vocab_size = getattr(config, 'vocab_size', 50000)
                
                # Примерная оценка параметров
                num_params = (
                    num_layers * (12 * hidden_size * hidden_size + 13 * hidden_size) +
                    vocab_size * hidden_size +
                    hidden_size * hidden_size  # embeddings
                )
            else:
                # Fallback: используем размер модели из HuggingFace
                try:
                    info = model_info(model_name)
                    if info.sha:
                        # Примерная оценка: 1B параметров = ~2GB (FP16)
                        num_params = 1_000_000_000  # консервативная оценка
                    else:
                        num_params = 7_000_000_000  # для 7B моделей
                except:
                    num_params = 7_000_000_000  # дефолт для 7B
        
        # Расчет памяти в зависимости от квантования
        if quantization_bits is None or quantization_bits == 0:
            # FP16: ~2 байта на параметр
            model_memory_gb = (num_params * 2.0) / (1024 ** 3)
        elif quantization_bits == 4:
            # 4-bit: ~0.5 байта на параметр
            model_memory_gb = (num_params * 0.5) / (1024 ** 3)
        elif quantization_bits == 8:
            # 8-bit: ~1 байт на параметр
            model_memory_gb = (num_params * 1.0) / (1024 ** 3)
        else:
            # FP16: ~2 байта на параметр
            model_memory_gb = (num_params * 2.0) / (1024 ** 3)
        
        return {
            'model_memory_gb': round(model_memory_gb, 2),
            'num_params': num_params,
            'quantization_bits': quantization_bits
        }
    except Exception as e:
        return {
            'error': str(e),
            'model_memory_gb': 0,
            'num_params': 0
        }


def estimate_training_memory(
    model_memory_gb,
    batch_size,
    max_length,
    gradient_accumulation_steps=1,
    use_gradient_checkpointing=True,
    use_8bit_optimizer=True
):
    """
    Оценивает общую память для обучения
    
    Args:
        model_memory_gb: Память модели в GB
        batch_size: Размер батча
        max_length: Максимальная длина последовательности
        gradient_accumulation_steps: Шаги накопления градиентов
        use_gradient_checkpointing: Использовать ли gradient checkpointing
        use_8bit_optimizer: Использовать ли 8-bit оптимизатор
    
    Returns:
        dict: Оценка памяти в GB
    """
    # Память для активаций (зависит от batch_size и max_length)
    # Примерная формула: batch_size * max_length * hidden_size * layers * 2 байта
    activation_memory_gb = (batch_size * max_length * 4096 * 32 * 2) / (1024 ** 3)
    
    if use_gradient_checkpointing:
        activation_memory_gb *= 0.3  # Экономия ~70%
    
    # Память для градиентов (примерно равна памяти модели)
    gradient_memory_gb = model_memory_gb * 0.5  # LoRA только часть параметров
    
    # Память для оптимизатора
    if use_8bit_optimizer:
        optimizer_memory_gb = model_memory_gb * 0.25  # 8-bit оптимизатор
    else:
        optimizer_memory_gb = model_memory_gb * 2.0  # FP16 оптимизатор
    
    # Память для данных (токенизированные данные)
    data_memory_gb = (batch_size * max_length * 4) / (1024 ** 3)  # int32 tokens
    
    total_memory_gb = (
        model_memory_gb +
        activation_memory_gb +
        gradient_memory_gb +
        optimizer_memory_gb +
        data_memory_gb +
        1.0  # Overhead
    )
    
    return {
        'model_memory_gb': round(model_memory_gb, 2),
        'activation_memory_gb': round(activation_memory_gb, 2),
        'gradient_memory_gb': round(gradient_memory_gb, 2),
        'optimizer_memory_gb': round(optimizer_memory_gb, 2),
        'data_memory_gb': round(data_memory_gb, 2),
        'overhead_gb': 1.0,
        'total_memory_gb': round(total_memory_gb, 2)
    }


def get_available_memory():
    """Получает доступную память GPU"""
    if not torch.cuda.is_available():
        return {
            'available_gb': 0,
            'total_gb': 0,
            'device_name': 'CPU'
        }
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = total_memory - allocated_memory
    
    return {
        'available_gb': round(available_memory / (1024 ** 3), 2),
        'total_gb': round(total_memory / (1024 ** 3), 2),
        'device_name': torch.cuda.get_device_name(device),
        'allocated_gb': round(allocated_memory / (1024 ** 3), 2)
    }


def check_memory_sufficiency(required_gb, available_gb, threshold=0.9):
    """
    Проверяет достаточность памяти
    
    Args:
        required_gb: Требуемая память в GB
        available_gb: Доступная память в GB
        threshold: Порог использования (0.9 = 90%)
    
    Returns:
        dict: Результат проверки
    """
    max_usable = available_gb * threshold
    sufficient = required_gb <= max_usable
    
    return {
        'sufficient': sufficient,
        'required_gb': round(required_gb, 2),
        'available_gb': round(available_gb, 2),
        'max_usable_gb': round(max_usable, 2),
        'recommendation': (
            '✅ Памяти достаточно' if sufficient
            else f'⚠️ Недостаточно памяти. Требуется: {required_gb:.2f} GB, доступно: {max_usable:.2f} GB'
        )
    }
