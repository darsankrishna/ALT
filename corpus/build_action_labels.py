# corpus/build_action_labels.py
import json
from pathlib import Path

UCF_CLASS_FILE  = "/tmp/ucf_splits/ucfTrainTestlist/classInd.txt"
HMDB_SPLITS_DIR = "/tmp/hmdb_splits/testTrainMultipleSplits"

# UCF-101: parse classInd.txt → {idx: name}  (0-indexed)
ucf_labels = {}
with open(UCF_CLASS_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx, name = line.split(" ", 1)
        ucf_labels[int(idx) - 1] = name.replace("_", " ").lower()

# HMDB-51: class name from filename prefix before "_test_split"
hmdb_labels = {}
for i, p in enumerate(sorted(Path(HMDB_SPLITS_DIR).glob("*_test_split1.txt"))):
    name = p.stem.replace("_test_split1", "").replace("_", " ")
    hmdb_labels[i] = name

# Deduplicated set (for corpus building — ~130 unique)
all_labels = list(ucf_labels.values()) + list(hmdb_labels.values())
seen, unique = set(), []
for l in all_labels:
    if l not in seen:
        seen.add(l)
        unique.append(l)

json.dump({"labels": unique}, open("corpus/action_labels.json", "w"), indent=2)
print(f"Unique corpus labels: {len(unique)}")   # ~130

# Ordered 152-label list: index == class id for loss computation
# UCF: indices 0–100, HMDB: indices 101–151
ordered = [""] * 152
for idx, name in ucf_labels.items():
    ordered[idx] = name
for idx, name in hmdb_labels.items():
    ordered[idx + 101] = name

# Sanity check
assert all(s != "" for s in ordered), "Gap in label list — check classInd.txt or split files"
json.dump(ordered, open("corpus/all_labels_ordered.json", "w"), indent=2)

# Save maps for build_ucf_hmdb_index.py
json.dump(ucf_labels, open("corpus/ucf_class_map.json", "w"), indent=2)
json.dump(hmdb_labels, open("corpus/hmdb_class_map.json", "w"), indent=2)

print(f"Ordered labels: {len(ordered)} (152 expected)")