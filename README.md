# Fine-Tuning Assistant

Visual application for fine-tuning language models using LoRA and BitsAndBytes quantization.

## Possibilities

- 🎨 **Visual interface** - convenient web-interface built on Gradio
- 🤗 **HuggingFace support** - load models directly from the HuggingFace Hub
- 📊 **Memory estimation** - predict memory requirements before starting training
- 🔧 **Quantization** - support for 4-bit and 8-bit quantization via BitsAndBytes
- 🛡️ **Error handling** - automatic memory error handling with recommendations
- 📁 **Dataset format** - support for JSONL format with `instruction` and `output` fields

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure that you have CUDA and PyTorch with GPU support installed.

## Usage

### Starting the app

```bash
python app.py
```

The application will open in your browser at `http://localhost:7860`

### Dataset format

The dataset must be in JSONL (JSON Lines) format, where each line is a JSON object:

```json
{"instruction": "User question", "input": "", "output": "Model answer"}
```

Example:
```json
{"instruction": "What is capitalism?", "input": "", "output": "Capitalism is an economic system..."}
```

### Training parameters

- **Model**: Model name from HuggingFace (e.g., `Qwen/Qwen2.5-7B-Instruct`) or a path to a local model
- **Dataset**: Upload a file in JSONL format
- **Quantization**: Select 4-bit (recommended), 8-bit, or no quantization
- **Batch Size**: Batch size (usually 1 for large models)
- **Gradient Accumulation**: Number of gradient accumulation steps
- **LoRA parameters**: Rank (r), Alpha, Dropout

### Memory Estimation

Before starting training, use the "Memory Estimation" tab to check VRAM sufficiency. The application will automatically estimate:
- Model memory
- Activation memory
- Gradient memory
- Optimizer memory
- Total required memory

### Memory Error Handling

The application automatically handles memory errors and provides recommendations:
- Decrease `batch_size`
- Increase `gradient_accumulation_steps`
- Decrease `max_length`
- Use quantization

## Project Structure

- `app.py` - Main application with GUI
- `trainer_module.py` - Module for model training
- `memory_estimator.py` - Module for memory estimation
- `train.py` - Original training script (kept for compatibility)
- `requirements.txt` - Project dependencies

## Usage Examples

### Training a Qwen model

1. Run `python app.py`
2. In the "Model Name" field, enter `Qwen/Qwen2.5-7B-Instruct`
3. Upload your dataset in JSONL format
4. Select training parameters
5. Click "Estimate Memory" to check
6. Click "Start Training"

### Using a local model

Instead of a HuggingFace ID, you can specify the path to a local model:
```
./my-local-model
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (minimum 8GB VRAM recommended)
- PyTorch with CUDA support

## Supported Models

The application supports any models based on Transformers, including:
- Qwen/Qwen2.5
- Llama 2/3
- Mistral
- Phi
- And other models with Transformer architecture

## License

MIT
