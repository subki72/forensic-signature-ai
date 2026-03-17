import requests
import json
import time

# API Endpoint (Ensure FastAPI server is running)
API_URL = "http://localhost:8000/verify"

def test_signature(master_path, test_path, test_description):
    print(f"\n[TESTING] {test_description}")
    print(f"Master File : {master_path}")
    print(f"Test File   : {test_path}")
    
    try:
        # Open both images in binary read mode
        with open(master_path, 'rb') as f_master, open(test_path, 'rb') as f_test:
            
            # Package as form-data
            files = {
                'file_asli': f_master,
                'file_uji': f_test
            }
            
            # Execute API request
            start_time = time.time()
            response = requests.post(API_URL, files=files)
            end_time = time.time()
            
            # Display Results
            if response.status_code == 200:
                result = response.json()
                print(f"Processing Time: {round(end_time - start_time, 2)} seconds")
                print("VERIFICATION RESULT:")
                print(json.dumps(result, indent=4))
            else:
                print(f"[ERROR] API Response {response.status_code}: {response.text}")
                
    except FileNotFoundError as e:
        print(f"[ERROR] Image file not found. Check the paths. ({e})")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Connection failed. Ensure 'uvicorn api.main:app' is running.")

# ==========================================
# TESTING SCENARIOS
# ==========================================
if __name__ == "__main__":
    master_signature = "data/processed/asli_master/asli_1.jpg"
    genuine_test_signature = "data/processed/asli_master/asli_2.jpg"
    
    # Sample from the Kaggle dataset
    forged_test_signature = "data/raw/signatures/full_org/original_1_1.png" 

    # 1. POSITIVE TEST (Expected: High Score, Verified)
    test_signature(master_signature, genuine_test_signature, "Genuine vs Genuine (Should be Verified)")
    
    # 2. NEGATIVE TEST (Expected: Low Score, Rejected)
    test_signature(master_signature, forged_test_signature, "Genuine vs Forged/Unseen (Should be Rejected)")