from fastapi import APIRouter
from fastapi.responses import FileResponse
from services.chatbot_service import process_message
import os, json
import uuid

router = APIRouter()

# Base directory of project (three levels up from backend)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Paths
REPORT_JSON_PATH = os.path.join(BASE_DIR, "results", "reports", "phase3_summary.json")
DASHBOARD_PRESENT_PATH = os.path.join(BASE_DIR, "results", "reports", "present_session_dashboard.html")
DASHBOARD_OVERALL_PATH = os.path.join(BASE_DIR, "results", "reports", "overall_sessions_dashboard.html")

# Chat endpoint
@router.post("/chat")
async def chat_endpoint(request: dict):
    message = request.get("message", "")
    session_id = request.get("session_id", str(uuid.uuid4()))  # fallback
    formatted_response, reply = process_message(message, session_id)
    return {
        "reply": reply,
        "formatted_response": formatted_response,
        "session_id": session_id
    }

# JSON summary
@router.get("/reports/json")
async def get_reports_json():
    if os.path.exists(REPORT_JSON_PATH):
        with open(REPORT_JSON_PATH, "r") as f:
            summary = json.load(f)
        return summary
    return {"error": "No JSON report available"}

# Present session dashboard
@router.get("/reports/present")
async def present_dashboard():
    if os.path.exists(DASHBOARD_PRESENT_PATH):
        return FileResponse(DASHBOARD_PRESENT_PATH, media_type="text/html")
    return {"error": "Present session dashboard not found"}

# Overall sessions dashboard
@router.get("/reports/overall")
async def overall_dashboard():
    if os.path.exists(DASHBOARD_OVERALL_PATH):
        return FileResponse(DASHBOARD_OVERALL_PATH, media_type="text/html")
    return {"error": "Overall sessions dashboard not found"}
