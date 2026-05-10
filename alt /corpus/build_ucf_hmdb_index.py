# corpus/build_ucf_hmdb_index.py
import json
from pathlib import Path

UCF_SPLIT  = "/tmp/ucf_splits/ucfTrainTestlist"
HMDB_SPLIT = "/tmp/hmdb_splits/testTrainMultipleSplits"

# UCF class name → 0-indexed int
ucf_class_map = {}
with open(f"{UCF_SPLIT}/classInd.txt") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx, name = line.split(" ", 1)
        ucf_class_map[name] = int(idx) - 1

def build_ucf(split_file, out_path):
    entries = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rel_path = parts[0]
            class_name = rel_path.split("/")[0]
            label = ucf_class_map[class_name]
            entries.append({"path": rel_path, "label": label, "dataset": "ucf"})
    json.dump(entries, open(out_path, "w"), indent=2)
    print(f"Saved {len(entries)} entries → {out_path}")

build_ucf(f"{UCF_SPLIT}/trainlist01.txt", "corpus/ucf_train_split1.json")
build_ucf(f"{UCF_SPLIT}/testlist01.txt",  "corpus/ucf_val_split1.json")

# HMDB class name → 0-indexed within HMDB
hmdb_classes = sorted({
    p.stem.replace("_test_split1", "")
    for p in Path(HMDB_SPLIT).glob("*_test_split1.txt")
})
hmdb_class_map = {name: i for i, name in enumerate(hmdb_classes)}

def build_hmdb(split_num, out_path, flag, label_offset=101):
    """flag=1 for train, flag=2 for test/val."""
    entries = []
    for p in sorted(Path(HMDB_SPLIT).glob(f"*_test_split{split_num}.txt")):
        class_name = p.stem.replace(f"_test_split{split_num}", "")
        label = hmdb_class_map[class_name] + label_offset
        with open(p) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                fname, file_flag = parts
                if int(file_flag) == flag:
                    entries.append({
                        "path": f"{class_name}/{fname}",
                        "label": label,
                        "dataset": "hmdb"
                    })
    json.dump(entries, open(out_path, "w"), indent=2)
    print(f"Saved {len(entries)} entries → {out_path}")

# CRITICAL: train uses flag=1, val/test uses flag=2
build_hmdb(split_num=1, out_path="corpus/hmdb_train_split1.json", flag=1)
build_hmdb(split_num=1, out_path="corpus/hmdb_val_split1.json",   flag=2)

# Combined train
ucf_train  = json.load(open("corpus/ucf_train_split1.json"))
hmdb_train = json.load(open("corpus/hmdb_train_split1.json"))
combined   = ucf_train + hmdb_train
json.dump(combined, open("corpus/train_labels.json", "w"), indent=2)
print(f"Combined train: {len(combined)} videos, 152 classes")

import shutil
shutil.copy("corpus/ucf_val_split1.json", "corpus/val_labels.json")