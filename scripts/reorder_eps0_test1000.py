import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SRC = "/home/stormai/userfile1/zhaoziru/compselect_data/extractor/hotpotqa_train_eps0_test1000.jsonl"
DST = "/home/stormai/userfile1/zhaoziru/compselect_data/extractor/hotpotqa_train_eps0_test1000_sorted.jsonl"

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL, device=device)


def sorted_sentences(query, sentences):
    if len(sentences) <= 1:
        return sentences

    q_emb = model.encode([query], convert_to_tensor=True)
    s_emb = model.encode(sentences, convert_to_tensor=True)

    sims = cosine_similarity(
        q_emb.cpu().numpy(),
        s_emb.cpu().numpy()
    )[0]

    return [
        s
        for s, _ in sorted(
            zip(sentences, sims),
            key=lambda x: x[1],
            reverse=True,
        )
    ]


count = 0
changed = 0

with open(SRC, encoding="utf-8") as fin, \
     open(DST, "w", encoding="utf-8") as fout:

    for line in fin:
        x = json.loads(line)

        old = x["sentences"]
        new = sorted_sentences(x["question"], old)

        if old != new:
            changed += 1

        x["sentences_original"] = old
        x["sentences"] = new

        fout.write(json.dumps(x, ensure_ascii=False) + "\n")
        count += 1

print("total:", count)
print("order changed:", changed)
print("changed ratio:", changed / count if count else 0)
print("saved:", DST)
