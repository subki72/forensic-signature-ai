import subprocess
import time
import sys

print("[1/2] Initializing Forensic AI Engine (FastAPI)...")
# sys.executable ensures the script uses the active environment's Python path
api_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--reload"])

print("Waiting 3 seconds for the API to stand by...")
time.sleep(3)

print("[2/2] Initializing Web Interface (Streamlit)...")
web_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"])

try:
    # Keep the script running to monitor both processes
    api_process.wait()
    web_process.wait()
except KeyboardInterrupt:
    # Graceful shutdown on Ctrl+C
    print("\n[STOP] Termination signal received. Shutting down all systems...")
    api_process.terminate()
    web_process.terminate()
    print("Systems successfully terminated. Goodbye.")