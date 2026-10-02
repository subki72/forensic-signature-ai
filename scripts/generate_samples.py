"""Generate synthetic signature image samples for local manual testing.

Creates 3 PNG image specimens in the `samples/` folder:
1. specimen_asli_1.png (Genuine reference specimen)
2. specimen_asli_2.png (Slight natural variant from the same signer)
3. specimen_palsu.png (Forged specimen with different angular strokes)
"""

import os
import cv2
import numpy as np


def create_signature_samples(output_dir: str = "samples") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 1. Specimen Asli 1 (Curved flow)
    canvas1 = np.ones((300, 600), dtype=np.uint8) * 255
    pts1 = [
        (50, 200), (90, 80), (140, 220), (190, 110),
        (240, 180), (300, 140), (380, 230), (450, 160),
        (520, 210), (550, 190)
    ]
    cv2.polylines(canvas1, [np.array(pts1, np.int32).reshape((-1, 1, 2))], False, 20, 3, cv2.LINE_AA)
    cv2.line(canvas1, (40, 245), (540, 240), 25, 2, cv2.LINE_AA)
    cv2.circle(canvas1, (555, 235), 3, 20, -1)
    p1 = os.path.join(output_dir, "specimen_asli_1.png")
    cv2.imwrite(p1, canvas1)

    # 2. Specimen Asli 2 (Natural variance, same signer)
    canvas2 = np.ones((300, 600), dtype=np.uint8) * 255
    pts2 = [
        (52, 198), (92, 83), (139, 218), (192, 112),
        (238, 182), (302, 138), (378, 228), (452, 162),
        (518, 208), (548, 192)
    ]
    cv2.polylines(canvas2, [np.array(pts2, np.int32).reshape((-1, 1, 2))], False, 22, 3, cv2.LINE_AA)
    cv2.line(canvas2, (42, 246), (538, 241), 25, 2, cv2.LINE_AA)
    cv2.circle(canvas2, (553, 236), 3, 22, -1)
    p2 = os.path.join(output_dir, "specimen_asli_2.png")
    cv2.imwrite(p2, canvas2)

    # 3. Specimen Palsu (Angular forgery)
    canvas3 = np.ones((300, 600), dtype=np.uint8) * 255
    pts3 = [
        (50, 120), (120, 240), (200, 70), (280, 250),
        (360, 80), (440, 240), (520, 100)
    ]
    cv2.polylines(canvas3, [np.array(pts3, np.int32).reshape((-1, 1, 2))], False, 15, 4, cv2.LINE_AA)
    p3 = os.path.join(output_dir, "specimen_palsu.png")
    cv2.imwrite(p3, canvas3)

    print(f"Sample signatures generated successfully in folder '{output_dir}/':")
    print(f" - {p1} (Tanda Tangan Asli Utama)")
    print(f" - {p2} (Tanda Tangan Asli Varian)")
    print(f" - {p3} (Tanda Tangan Palsu/Berbeda)")


if __name__ == "__main__":
    create_signature_samples()
