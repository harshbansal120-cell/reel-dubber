"""Start the app: python run_app.py"""

import uvicorn

if __name__ == "__main__":
    print("VYOMA is starting at http://127.0.0.1:8000")
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, log_level="warning")
