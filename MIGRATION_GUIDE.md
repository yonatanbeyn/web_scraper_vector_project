# Migration Guide - OpenAI API Update

## What Changed

Due to dependency conflicts, I've updated both scripts to use the **modern OpenAI API (v1+)** instead of the legacy v0.28 API.

### API Syntax Changes

#### Old API (v0.28)
```python
import openai
openai.api_key = OPENAI_API_KEY

response = openai.Embedding.create(
    input=chunk,
    model="text-embedding-3-small"
)
vector = response['data'][0]['embedding']
```

#### New API (v1.109.1+)
```python
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

response = client.embeddings.create(
    input=chunk,
    model="text-embedding-3-small"
)
vector = response.data[0].embedding
```

## Files Updated

1. **requirements.txt** - Changed `openai==0.28` to `openai>=1.109.1`
2. **scrape_and_store.py** - Updated to use new OpenAI client syntax
3. **scrape_and_store_langchain.py** - Already uses new API via LangChain

## Installation Instructions

### Clean Install (Recommended)

```bash
# 1. Deactivate and remove old virtual environment
deactivate
rm -rf venv

# 2. Create fresh virtual environment
python3.13 -m venv venv
source venv/bin/activate

# 3. Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Upgrade Existing Environment

```bash
# Activate your existing environment
source venv/bin/activate

# Uninstall old OpenAI version
pip uninstall openai -y

# Install all dependencies (will install new OpenAI)
pip install -r requirements.txt
```

## Verification

Check that both OpenAI and LangChain are installed correctly:

```bash
pip list | grep -E "(openai|langchain)"
```

Expected output:
```
langchain                 0.x.x
langchain-community       0.x.x
langchain-openai          1.1.9
langchain-text-splitters  0.x.x
openai                    1.109.1 (or higher)
```

## Running the Scripts

Both scripts now work with the same dependencies!

### Original Script (Updated)
```bash
python scrape_and_store.py
```

### LangChain Version
```bash
python scrape_and_store_langchain.py
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'openai'"
**Solution:**
```bash
pip install openai>=1.109.1
```

### Issue: "AttributeError: module 'openai' has no attribute 'Embedding'"
**Solution:** You're using old syntax with new API. Make sure you're running the updated scripts.

### Issue: Dependency conflict errors
**Solution:** Use clean install method above to avoid conflicts.

## Benefits of New API

1. **Type Safety** - Better IDE autocomplete and type hints
2. **Streaming Support** - Native streaming for completions
3. **Better Error Handling** - More informative error messages
4. **Modern Patterns** - Follows current Python best practices
5. **LangChain Compatible** - Works seamlessly with LangChain ecosystem

## Need the Old Version?

If you absolutely need the old API v0.28, you can:

1. Create a separate environment for the original script
2. Use the LangChain version instead (recommended)

The LangChain version provides all the same functionality with better abstraction!
