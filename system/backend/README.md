# Backend

FastAPI backend for the DyRIFT inference system.

## Run

```powershell
cd system/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check:

```text
GET http://127.0.0.1:8000/api/health
```

The first version stores data in SQLite under `system/backend/data/`.
