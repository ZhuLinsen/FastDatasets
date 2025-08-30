# 🚀 FastDatasets – HF Spaces Demo

Transform documents into LLM training datasets with smart fallback support!

## Features
- **Smart Processing**: Tries full FastDatasets, falls back to simplified OpenAI calls
- **Cost Control**: Max 2000 chars processed, 2-5 QA pairs output
- **Clean Format**: Always outputs Alpaca-format JSON
- **Error Resilient**: Handles import failures gracefully

## Setup Secrets
In Space Settings → Repository secrets, add:

**Required:**
- `LLM_API_KEY` - Your OpenAI/DeepSeek/etc API key

**Optional:**
- `LLM_API_BASE` - API endpoint (e.g., `https://api.deepseek.com/v1`)
- `LLM_MODEL` - Model name (e.g., `deepseek-chat`, `gpt-3.5-turbo`)

⚠️ **Important**: Ensure no trailing spaces/newlines in secret values!

## How it works
1. **Primary Mode**: Uses full FastDatasets if package installs successfully
2. **Fallback Mode**: Uses simplified OpenAI client if FastDatasets fails
3. **Demo Mode**: Shows precomputed examples

## Files Structure
```
app.py                          # Main Gradio app
requirements.txt               # Dependencies  
samples/mini.txt              # Sample input
samples/precomputed/dataset-alpaca.json  # Example output
```

## Troubleshooting
- **URL errors**: Check for whitespace in `LLM_API_BASE`
- **Import failures**: Fallback mode will activate automatically
- **API errors**: Verify `LLM_API_KEY` and `LLM_API_BASE` are correct