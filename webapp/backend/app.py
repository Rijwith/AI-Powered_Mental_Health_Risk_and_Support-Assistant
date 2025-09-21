# webapp/backend/app.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import chat
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Mental Health Chatbot API")

# Allow frontend (React) to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "Mental Health Chatbot Backend Running"}

# -----------------------------
# Serve reports folder as static
# -----------------------------
# Absolute path to the reports folder
reports_dir = r"C:\Users\rijju\Desktop\mental-health-chatbot\results\reports"

# Mount only if it exists
if os.path.exists(reports_dir) and os.path.isdir(reports_dir):
    app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")
    print(f"[Info] Reports mounted at /reports from {reports_dir}")
else:
    print(f"[Warning] Reports directory does not exist: {reports_dir}")

