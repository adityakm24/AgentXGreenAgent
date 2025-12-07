import uvicorn
import os
from pathlib import Path
from src.db import init_db

def main():
    print("Initializing Jurist-Bench Environment...")
    
    # Ensure DB exists
    if not Path("data/jurist.db").exists():
        print("Database not found. Initializing...")
        init_db()
        # Note: If no data, ingest.py should be run. 
        # We can automate this or just warn.
        print("WARNING: Database is empty. Please run 'python -m src.ingest' to load data.")
    
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Green Agent Server on port {port}...")
    
    # Run the server
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    main()
