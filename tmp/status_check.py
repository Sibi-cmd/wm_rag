import httpx, sys
try:
    r = httpx.get('http://127.0.0.1:8000/status', timeout=5.0)
    print('STATUS', r.status_code)
    print('BODY', r.text)
except Exception as e:
    print('ERROR', e, file=sys.stderr)
    sys.exit(1)
