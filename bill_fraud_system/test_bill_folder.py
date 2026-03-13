import os
import glob
import requests
import json
import time

FOLDER_PATH = "data/Bill"
URL = "http://localhost:8000/api/analyze"

def main():
    print(f"--- Running Fraud Engine on {FOLDER_PATH} ---")
    
    # Get all files (pdfs, jpgs, pngs)
    files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.pdf']:
        files.extend(glob.glob(os.path.join(FOLDER_PATH, ext)))
        # case insensitive ext match
        files.extend(glob.glob(os.path.join(FOLDER_PATH, ext.upper())))
    
    # Deduplicate
    files = list(set(files))
    
    if not files:
        print(f"No files found in {FOLDER_PATH}")
        return
        
    print(f"Found {len(files)} files to analyze. Starting...\n")
    
    results = {"GENUINE": 0, "TAMPERED": 0, "FAILED": 0}
    
    start_time = time.time()
    for i, file_path in enumerate(sorted(files)):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'rb') as f:
                response = requests.post(
                    URL, 
                    files={"file": (filename, f)},
                    data={"doc_type": "bill"}
                )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "UNKNOWN")
                conf = data.get("confidence", 0)
                reason = data.get("tamper_reason", "")
                
                results[status] = results.get(status, 0) + 1
                
                icon = "✅" if status == "GENUINE" else "⚠️"
                reason_str = f" | Reason: {reason}" if status == "TAMPERED" else ""
                print(f"[{i+1}/{len(files)}] {icon} {filename:<45} {status:<8} ({conf}%) {reason_str}")
            else:
                results["FAILED"] += 1
                print(f"[{i+1}/{len(files)}] ❌ {filename:<45} FAILED (HTTP {response.status_code}): {response.text}")
                
        except Exception as e:
            results["FAILED"] += 1
            print(f"[{i+1}/{len(files)}] ❌ {filename:<45} ERROR: {str(e)}")
            
    total_time = time.time() - start_time
    print(f"\n--- Summary ---")
    print(f"Total Files: {len(files)}")
    print(f"GENUINE:   {results.get('GENUINE', 0)}")
    print(f"TAMPERED:  {results.get('TAMPERED', 0)}")
    print(f"FAILED:    {results.get('FAILED', 0)}")
    print(f"Time taken: {total_time:.2f} seconds ({total_time/len(files):.2f}s per file)")

if __name__ == "__main__":
    main()
