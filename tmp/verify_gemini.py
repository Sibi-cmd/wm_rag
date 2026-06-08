import os, sys, traceback

def report(step, success, detail=''):
    print(f'{step}:', 'PASS' if success else 'FAIL', f'| {detail}')

# 1. import google.generativeai
try:
    import google.generativeai as genai
    report('Import google.generativeai', True)
except Exception as e:
    report('Import google.generativeai', False, str(e))
    genai = None

# 2. GEMINI_API_KEY loading
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    report('GEMINI_API_KEY loading', True)
else:
    report('GEMINI_API_KEY loading', False, 'Env var not set')

# Configure if possible
if genai and api_key:
    try:
        genai.configure(api_key=api_key)
        report('Gemini configure', True)
    except Exception as e:
        report('Gemini configure', False, str(e))
else:
    report('Gemini configure', False, 'Skipped')

# 3. GeminiClient initialization
try:
    from app.llm_client import GeminiClient
    client = GeminiClient()
    report('GeminiClient init', True, f'model={client.model_name}')
except Exception as e:
    report('GeminiClient init', False, str(e))
    client = None

# 4. RAGPipeline initialization
try:
    from app.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    report('RAGPipeline init', True)
except Exception as e:
    report('RAGPipeline init', False, str(e))
    pipeline = None

# 5. FastAPI startup (instantiate app)
try:
    from app.main import app as fastapi_app
    report('FastAPI app import', True)
except Exception as e:
    report('FastAPI app import', False, str(e))

print('Verification completed')
