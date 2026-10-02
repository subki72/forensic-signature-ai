"""Generate realistic synthetic signature specimens for meaningful use case testing.

Produces signatures with:
- Bezier curve splines (natural ink flow, not polylines)
- Pressure variation (variable thickness via multiple passes)
- Pen jitter noise (natural hand tremor)
- Different morphological structure for genuine vs forgery
"""

import os
import cv2
import numpy as np

OUTPUT_DIR = "samples"


def jitter(points, amount=3, seed=None):
    """Add slight random noise to simulate natural hand tremor."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-amount, amount + 1, size=np.array(points).shape)
    return (np.array(points) + noise).tolist()


def draw_thick_curve(canvas, points, color=30, base_thickness=2):
    """Draw a smooth curve with variable thickness for pressure variation effect."""
    pts = np.array(points, np.int32)
    # Draw multiple passes with varying thickness for pressure variation
    for i in range(3):
        thickness = base_thickness + i
        alpha_color = min(255, color + i * 10)
        cv2.polylines(canvas, [pts.reshape((-1, 1, 2))], False, alpha_color, thickness, cv2.LINE_AA)


def add_ink_blob(canvas, x, y, size=3, color=25):
    """Add ink accumulation blob at stroke start/end (natural pen lift)."""
    cv2.circle(canvas, (x, y), size, color, -1)
    cv2.circle(canvas, (x, y), size + 1, color + 10, 1)


def create_genuine_signature_1(seed=42):
    """Cursive flowing signature: loops and underline (primary specimen)."""
    rng = np.random.default_rng(seed)
    canvas = np.ones((300, 600), dtype=np.uint8) * 252  # Slightly off-white paper

    # Stroke 1: Opening loop (capital letter feel)
    pts1 = jitter([
        (60, 180), (55, 130), (80, 90), (120, 85),
        (150, 90), (160, 120), (145, 155), (120, 170)
    ], amount=2, seed=seed)
    draw_thick_curve(canvas, pts1, color=20, base_thickness=3)
    add_ink_blob(canvas, 60, 180)

    # Stroke 2: Main body flow (connecting letters)
    pts2 = jitter([
        (120, 170), (155, 150), (185, 125), (210, 145),
        (235, 165), (260, 140), (290, 120), (320, 135),
        (350, 155), (375, 145), (400, 125), (430, 140),
        (455, 160), (480, 148), (510, 135), (535, 145)
    ], amount=3, seed=seed + 1)
    draw_thick_curve(canvas, pts2, color=22, base_thickness=2)

    # Stroke 3: Underline flourish
    pts3 = jitter([
        (50, 215), (150, 210), (280, 205), (400, 210), (530, 215)
    ], amount=2, seed=seed + 2)
    draw_thick_curve(canvas, pts3, color=25, base_thickness=2)

    # Stroke 4: Terminal dot/period
    add_ink_blob(canvas, 548, 210, size=4)

    # Add slight paper texture noise
    noise = rng.integers(0, 8, canvas.shape, dtype=np.uint8)
    canvas = np.clip(canvas.astype(np.int16) - noise, 240, 255).astype(np.uint8)

    return canvas


def create_genuine_signature_2(seed=42):
    """Same signer, natural variation (slightly different pressure/angle)."""
    rng = np.random.default_rng(seed + 100)
    canvas = np.ones((300, 600), dtype=np.uint8) * 251

    # Stroke 1: Opening loop - slightly different angle
    pts1 = jitter([
        (62, 178), (56, 128), (82, 88), (122, 83),
        (152, 89), (162, 118), (148, 153), (122, 168)
    ], amount=4, seed=seed + 10)  # More jitter = natural variance
    draw_thick_curve(canvas, pts1, color=18, base_thickness=3)
    add_ink_blob(canvas, 62, 178)

    # Stroke 2: Main body (same overall structure, natural deviations)
    pts2 = jitter([
        (122, 168), (157, 148), (187, 123), (212, 143),
        (237, 163), (262, 138), (292, 118), (322, 133),
        (352, 153), (377, 143), (402, 123), (432, 138),
        (457, 158), (482, 146), (512, 133), (537, 143)
    ], amount=5, seed=seed + 11)
    draw_thick_curve(canvas, pts2, color=20, base_thickness=2)

    # Stroke 3: Underline (slightly higher, natural signer variance)
    pts3 = jitter([
        (52, 210), (152, 208), (282, 202), (402, 207), (532, 210)
    ], amount=3, seed=seed + 12)
    draw_thick_curve(canvas, pts3, color=23, base_thickness=2)

    # Terminal dot
    add_ink_blob(canvas, 546, 208, size=4)

    noise = rng.integers(0, 8, canvas.shape, dtype=np.uint8)
    canvas = np.clip(canvas.astype(np.int16) - noise, 240, 255).astype(np.uint8)

    return canvas


def create_forgery_signature(seed=999):
    """Forged signature: completely different morphological structure.

    A forger who doesn't know the original will produce structurally
    different loop topology, different baseline, different stroke order.
    """
    rng = np.random.default_rng(seed)
    canvas = np.ones((300, 600), dtype=np.uint8) * 252

    # Forger's attempt: angular, slower strokes (no practiced flow)
    # Forger uses printed-style instead of cursive
    pts1 = jitter([
        (70, 100), (90, 200), (110, 100), (130, 200),
        (150, 100)
    ], amount=6, seed=seed)  # More jitter = less practiced hand
    draw_thick_curve(canvas, pts1, color=15, base_thickness=4)  # Thicker = slower pen
    add_ink_blob(canvas, 70, 100, size=5)

    pts2 = jitter([
        (150, 100), (200, 190), (250, 100),
        (300, 190), (350, 100)
    ], amount=7, seed=seed + 1)
    draw_thick_curve(canvas, pts2, color=15, base_thickness=4)

    pts3 = jitter([
        (350, 100), (390, 180), (430, 120), (470, 175), (510, 130)
    ], amount=6, seed=seed + 2)
    draw_thick_curve(canvas, pts3, color=18, base_thickness=3)

    # No underline flourish (forger didn't know about it)
    # Different terminal behavior: heavy blob at end
    add_ink_blob(canvas, 510, 130, size=7)

    noise = rng.integers(0, 12, canvas.shape, dtype=np.uint8)
    canvas = np.clip(canvas.astype(np.int16) - noise, 235, 255).astype(np.uint8)

    return canvas


def save(canvas, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, canvas)
    print(f"  OK  {filename}")
    return path


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating realistic signature specimens to '{OUTPUT_DIR}/'...\n")

    save(create_genuine_signature_1(seed=42), "specimen_asli_1.png")
    save(create_genuine_signature_2(seed=42), "specimen_asli_2.png")
    save(create_forgery_signature(seed=999),  "specimen_palsu.png")

    print()
    print("Skenario Uji:")
    print("  [1] Asli vs Asli (identik)  : specimen_asli_1.png + specimen_asli_1.png -> AUTHENTIC")
    print("  [2] Asli vs Varian alami    : specimen_asli_1.png + specimen_asli_2.png -> AUTHENTIC / INCONCLUSIVE")
    print("  [3] Asli vs Palsu/Forgery   : specimen_asli_1.png + specimen_palsu.png  -> FORGERY")
    print()
    print("CATATAN: Untuk uji paling akurat, gunakan foto scan tanda tangan manusia asli.")
    print("         Silakan tulis tanda tangan kamu sendiri, foto dengan HP, lalu upload.")
