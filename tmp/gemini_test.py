import sys, os, asyncio
sys.path.append('d:/rag/wm_rag')
from app.llm_client import GeminiClient

async def main():
    client = GeminiClient()
    print('Gemini client initialized, model:', client.model_name)
    try:
        resp = await client.generate_response('Say hello')
        print('Response snippet:', resp[:100])
    except Exception as e:
        print('Error during generation:', e)

if __name__ == '__main__':
    asyncio.run(main())
