import json
import csv
import os

src = os.path.expanduser(
    "~/userfile1/zhaoziru/compselect_data/raw/hotpotqa_train.jsonl"
)

dst = os.path.expanduser(
    "~/userfile1/zhaoziru/compselect_data/raw/hotpotqa_train_dpr.tsv"
)

with open(src, "r", encoding="utf-8") as fin, \
     open(dst, "w", encoding="utf-8", newline="") as fout:

    writer = csv.writer(fout, delimiter="\t")

    count = 0

    for line in fin:
        x = json.loads(line)

        question = x["question"]
        answer = x["answer"]

        # DPR CsvQASrc expects Python-list syntax in column 2
        writer.writerow([
            question,
            repr([answer])
        ])

        count += 1

print("saved:", dst)
print("samples:", count)
