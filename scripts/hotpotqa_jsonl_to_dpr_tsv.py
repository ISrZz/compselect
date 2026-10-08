import json
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

count = 0

with open(args.input, "r", encoding="utf-8") as fin, \
     open(args.output, "w", encoding="utf-8", newline="") as fout:

    writer = csv.writer(fout, delimiter="\t")

    for line in fin:
        line = line.strip()
        if not line:
            continue

        x = json.loads(line)

        question = x["question"]
        answer = x["answer"]

        # DPR CsvQASrc expects a Python list in the second column
        writer.writerow([
            question,
            repr([answer])
        ])

        count += 1

print("saved:", args.output)
print("samples:", count)
