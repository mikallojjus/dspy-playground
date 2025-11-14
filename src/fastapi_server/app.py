import os
import uvicorn

from fastapi import FastAPI
from routers import (
    guest_extraction_router,
)
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.include_router(guest_extraction_router, prefix="/api", tags=["guest_extraction"])

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
