from datasets import load_dataset
import json
import os

out_dir = os.path.expanduser(
    "~/userfile1/zhaoziru/compselect_data/raw"
)
os.makedirs(out_dir, exist_ok=True)

# -----------------------
# HotpotQA
# -----------------------
ds = load_dataset("hotpotqa/hotpot_qa", "distractor")

with open(os.path.join(out_dir, "hotpotqa_train.jsonl"), "w") as f:
    for x in ds["train"]:
        item = {
            "question": x["question"],
            "answer": x["answer"],
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
