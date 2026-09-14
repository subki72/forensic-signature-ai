"""Dataset management and Triplet Loss synchronization for forensic notebooks."""

import json
from pathlib import Path

nb_dir = Path(__file__).resolve().parent.parent / "notebooks"

# ==============================================================================
# 1. Update 01_thresholding_test.ipynb
# ==============================================================================
p1 = nb_dir / "01_thresholding_test.ipynb"
with open(p1, "r", encoding="utf-8") as f:
    nb1 = json.load(f)

for cell in nb1["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "image_path =" in src:
            new_c7 = (
                "# Test computer vision preprocessing on genuine signature specimen\n"
                "image_path = '../extract/001/original_1_1.jpg'\n"
                "image = cv2.imread(image_path)\n"
                "\n"
                "if image is not None:\n"
                "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                "    blurred = cv2.GaussianBlur(gray, (5, 5), 0)\n"
                "    binary = cv2.adaptiveThreshold(\n"
                "        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2\n"
                "    )\n"
                "    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n"
                "    image_with_box = image.copy()\n"
                "    count = 0\n"
                "    for cnt in contours:\n"
                "        x, y, w, h = cv2.boundingRect(cnt)\n"
                "        if cv2.contourArea(cnt) > 200:\n"
                "            count += 1\n"
                "            cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)\n"
                "    display_image('Isolated Signature Strokes', image_with_box)\n"
                "    print(f'Signature strokes detected: {count} contours.')\n"
                "else:\n"
                "    print(f'Could not load {image_path}')"
            )
            cell["source"] = [l + "\n" for l in new_c7.splitlines()]
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

        if "anchor_dna =" in src:
            new_c9 = (
                "# Load genuine signatures for Signer 1 and skilled forgeries from extract/\n"
                "asli_files = sorted(glob.glob('../extract/001/original_1_*.jpg'))\n"
                "forgery_files = sorted(glob.glob('../extract/001_forg/forgeries_1_*.jpg'))\n"
                "\n"
                "results = []\n"
                "anchor_dna = extract_dna_from_path(asli_files[0])\n"
                "anchor_dna_norm = F.normalize(anchor_dna.unsqueeze(0), p=2, dim=1)\n"
                "\n"
                "# Compare Genuine vs. Genuine (Positive Pairs)\n"
                "gen_distances = []\n"
                "for i in range(1, len(asli_files)):\n"
                "    test_dna = extract_dna_from_path(asli_files[i])\n"
                "    test_dna_norm = F.normalize(test_dna.unsqueeze(0), p=2, dim=1)\n"
                "    sim = F.cosine_similarity(anchor_dna_norm, test_dna_norm).item()\n"
                "    dist = 1.0 - sim\n"
                "    gen_distances.append(dist)\n"
                "    results.append({'Type': 'Genuine vs Genuine (Pos)', 'Similarity': sim, 'Distance': dist})\n"
                "\n"
                "# Compare Genuine vs. Forged (Hard Negative Pairs)\n"
                "forg_distances = []\n"
                "for f in forgery_files:\n"
                "    test_dna = extract_dna_from_path(f)\n"
                "    test_dna_norm = F.normalize(test_dna.unsqueeze(0), p=2, dim=1)\n"
                "    sim = F.cosine_similarity(anchor_dna_norm, test_dna_norm).item()\n"
                "    dist = 1.0 - sim\n"
                "    forg_distances.append(dist)\n"
                "    results.append({'Type': 'Genuine vs Forged (Neg)', 'Similarity': sim, 'Distance': dist})\n"
                "\n"
                "df = pd.DataFrame(results)\n"
                "print('\\nFORENSIC BASELINE AUDIT SUMMARY (Cosine Similarity & Distance):')\n"
                "print(df.groupby('Type')[['Similarity', 'Distance']].describe()[[(c, s) for c in ['Similarity', 'Distance'] for s in ['mean', 'min', 'max']]])\n"
                "\n"
                "avg_pos_dist = np.mean(gen_distances) if gen_distances else 0.0\n"
                "avg_neg_dist = np.mean(forg_distances) if forg_distances else 0.0\n"
                "triplet_gap = avg_neg_dist - avg_pos_dist\n"
                "print(f'\\nBaseline Triplet Margin Gap [d(A, N) - d(A, P)]: {triplet_gap:.4f}')\n"
                "if triplet_gap < 0.3:\n"
                "    print('Diagnosis: The baseline model does NOT have sufficient margin separation for skilled forgeries.')\n"
                "    print('Action: Triplet Loss fine-tuning is REQUIRED to enforce margin separation >= 0.40.')"
            )
            cell["source"] = [l + "\n" for l in new_c9.splitlines()]
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open(p1, "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=1, ensure_ascii=False)
print("Updated 01_thresholding_test.ipynb successfully.")


# ==============================================================================
# 2. Update 02_train_siamese.ipynb
# ==============================================================================
p2 = nb_dir / "02_train_siamese.ipynb"
with open(p2, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

for cell in nb2["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "kaggle_raw =" in src or "kaggle_cropped" in src:
            new_c7 = (
                "os.makedirs('../data/processed/kaggle_cropped', exist_ok=True)\n"
                "# Load external signers directly from extract/\n"
                "kaggle_raw = sorted(glob.glob('../extract/*/original_*.jpg'))[:100]\n"
                "\n"
                "count = 0\n"
                "for path in kaggle_raw:\n"
                "    img = cv2.imread(path)\n"
                "    if img is None:\n"
                "        continue\n"
                "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n"
                "    blurred = cv2.GaussianBlur(gray, (5, 5), 0)\n"
                "    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)\n"
                "    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n"
                "    if contours:\n"
                "        c = max(contours, key=cv2.contourArea)\n"
                "        x, y, w, h = cv2.boundingRect(c)\n"
                "        pad = 15\n"
                "        x_p, y_p = max(0, x - pad), max(0, y - pad)\n"
                "        w_p, h_p = min(img.shape[1] - x_p, w + (pad * 2)), min(img.shape[0] - y_p, h + (pad * 2))\n"
                "        crop = img[y_p:y_p + h_p, x_p:x_p + w_p]\n"
                "        cv2.imwrite(f'../data/processed/kaggle_cropped/k_{count}.jpg', crop)\n"
                "        count += 1\n"
                "\n"
                "print(f'Successfully cropped and standardized {count} external signatures.')"
            )
            cell["source"] = [l + "\n" for l in new_c7.splitlines()]
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

        if "triplets = []" in src or "image_pairs = []" in src:
            new_c9 = (
                "# Load Signer 1 genuine signatures, skilled forgeries, and cropped external signatures from extract/\n"
                "asli_paths = sorted(glob.glob('../extract/001/original_1_*.jpg'))\n"
                "forgery_paths = sorted(glob.glob('../extract/001_forg/forgeries_1_*.jpg'))\n"
                "kaggle_paths = sorted(glob.glob('../data/processed/kaggle_cropped/*.jpg'))\n"
                "\n"
                "triplets = []\n"
                "\n"
                "# Generate triplets: (Anchor: Genuine, Positive: Genuine, Negative: Skilled/External)\n"
                "for i in range(len(asli_paths)):\n"
                "    for j in range(len(asli_paths)):\n"
                "        if i == j:\n"
                "            continue\n"
                "        anchor_path = asli_paths[i]\n"
                "        positive_path = asli_paths[j]\n"
                "        \n"
                "        # Hard negatives: skilled forgeries of Signer 1\n"
                "        for forg_path in forgery_paths:\n"
                "            triplets.append((anchor_path, positive_path, forg_path))\n"
                "            \n"
                "        # Random negatives: external signatures\n"
                "        if kaggle_paths:\n"
                "            sampled_ext = random.sample(kaggle_paths, min(3, len(kaggle_paths)))\n"
                "            for ext_path in sampled_ext:\n"
                "                triplets.append((anchor_path, positive_path, ext_path))\n"
                "\n"
                "train_dataset = TripletDataset(triplets, transform=train_transform)\n"
                "train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)\n"
                "\n"
                "print(f'Dataset generated with {len(triplets)} verification triplets.')"
            )
            cell["source"] = [l + "\n" for l in new_c9.splitlines()]
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open(p2, "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=1, ensure_ascii=False)
print("Updated 02_train_siamese.ipynb successfully.")
