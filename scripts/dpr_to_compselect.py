import json
import os

src = "/home/stormai/userfile1/zhaoziru/compselect_data/retrieved/hotpotqa_train_top5.json"

dst = "/home/stormai/userfile1/zhaoziru/compselect_data/retrieved/hotpotqa_train_top5_compselect.jsonl"
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

count = 0

with open(dst, "w", encoding="utf-8") as fout:
    for x in data:

        answers = x.get("answers", [])
        answer = answers[0] if answers else ""

        # 保留 title + text，信息更完整
        documents = []

        for ctx in x["ctxs"]:
            title = ctx.get("title", "").strip()
            text = ctx.get("text", "").strip()

            if title:
                doc = f"{title}\n{text}"
            else:
                doc = text

            documents.append(doc)

        item = {
            "question": x["question"],
            "answer": answer,
            "documents": documents
        }

        fout.write(
            json.dumps(item, ensure_ascii=False) + "\n"
        )

        count += 1

print("saved:", dst)
print("samples:", count)
