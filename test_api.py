"""
test_api.py
===========
Automated local test script for the /verify endpoint.
Runs positive (genuine vs. genuine) and negative (genuine vs. forged)
scenarios and prints the JSON response with timing information.

Usage:
    Ensure the FastAPI server is running first:
        uvicorn api.main:app --reload

    Then execute:
        python test_api.py
"""

import requests
import json
import time

API_URL = "http://localhost:8000/verify"


def test_signature(master_path: str, test_path: str, description: str) -> None:
    """Send two signature images to the verification API and print the result.

    Args:
        master_path: Filesystem path to the reference (genuine) signature.
        test_path: Filesystem path to the questioned signature.
        description: Human-readable label for this test scenario.
    """
    print(f"\n[TEST] {description}")
    print(f"  Reference : {master_path}")
    print(f"  Questioned: {test_path}")

    try:
        with open(master_path, "rb") as f_master, open(test_path, "rb") as f_test:
            files = {
                "file_asli": f_master,
                "file_uji": f_test,
            }

            start = time.time()
            response = requests.post(API_URL, files=files)
            elapsed = round(time.time() - start, 2)

            if response.status_code == 200:
                print(f"  Time: {elapsed}s")
                print("  Result:")
                print(json.dumps(response.json(), indent=4))
            else:
                print(f"  [ERROR] HTTP {response.status_code}: {response.text}")

    except FileNotFoundError as e:
        print(f"  [ERROR] File not found: {e}")
    except requests.exceptions.ConnectionError:
        print("  [ERROR] Connection refused. Is the API server running?")


if __name__ == "__main__":
    # Paths relative to the project root
    master = "data/processed/asli_master/asli_1.jpg"
    genuine = "data/processed/asli_master/asli_2.jpg"
    forged = "data/raw/signatures/full_forg/forgeries_1_1.png"

    # Positive test: two genuine specimens from the same signer
    test_signature(master, genuine, "Genuine vs. Genuine (expected: VERIFIED)")

    # Negative test: genuine specimen vs. a known forgery
    test_signature(master, forged, "Genuine vs. Forged (expected: REJECTED)")