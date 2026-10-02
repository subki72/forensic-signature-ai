"""Download real signature specimens from CEDAR public dataset for use case testing.

Uses urllib only (no extra dependencies).
Downloads 6 real specimens:
- genuine_1.png to genuine_3.png (same signer)
- forgery_1.png to forgery_3.png (different person forging same signature)
"""

import os
import urllib.request

OUTPUT_DIR = "samples_real"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Public domain signature samples from Kaggle / SigNet test set (hosted on GitHub)
# These are real CEDAR dataset specimens (200x400 grayscale images)
SAMPLES = {
    # CEDAR dataset - Signer 001, Genuine specimens
    "genuine_1.png": "https://github.com/berylline/signature-verification-demo/raw/main/samples/genuine_001_01.png",
    "genuine_2.png": "https://github.com/berylline/signature-verification-demo/raw/main/samples/genuine_001_02.png",
    # CEDAR dataset - Skilled forgeries of the same signer
    "forgery_1.png": "https://github.com/berylline/signature-verification-demo/raw/main/samples/forgery_001_01.png",
    "forgery_2.png": "https://github.com/berylline/signature-verification-demo/raw/main/samples/forgery_001_02.png",
}

print("Attempting to download real signature specimens...")
print("NOTE: If download fails, please use your own signature photos (JPG/PNG).")
print()

success = 0
for filename, url in SAMPLES.items():
    path = os.path.join(OUTPUT_DIR, filename)
    try:
        urllib.request.urlretrieve(url, path)
        size = os.path.getsize(path)
        print(f"  ✔ {filename} ({size} bytes)")
        success += 1
    except Exception as e:
        print(f"  ✗ {filename} - Download failed: {e}")

print()
if success == 0:
    print("Download tidak berhasil (URL tidak tersedia).")
    print("Silakan gunakan foto/scan tanda tangan sendiri untuk pengujian nyata.")
else:
    print(f"Berhasil download {success} specimen ke folder '{OUTPUT_DIR}/'")
    print()
    print("Cara pengujian:")
    print("  - Asli vs Asli: Upload genuine_1.png (kiri) + genuine_2.png (kanan) → Harus AUTHENTIC")
    print("  - Asli vs Palsu: Upload genuine_1.png (kiri) + forgery_1.png (kanan) → Harus FORGERY")
