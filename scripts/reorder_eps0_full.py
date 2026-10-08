import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SRC = "/home/stormai/userfile1/zhaoziru/compselect_data/extractor/hotpotqa_train_eps0.jsonl"
DST = "/home/stormai/userfile1/zhaoziru/compselect_data/extractor/hotpotqa_train_eps0_reordered.jsonl"

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)
print("loading:", MODEL)

model = SentenceTransformer(MODEL, device=device)


def sorted_sentences(query, sentences):
    if len(sentences) <= 1:
        return list(sentences)

    q_emb = model.encode(
        [query],
        convert_to_tensor=True
    )

    s_emb = model.encode(
        sentences,
        convert_to_tensor=True
    )

    similarities = cosine_similarity(
        q_emb.cpu().numpy(),
        s_emb.cpu().numpy()
    )[0]

    return [
        s
        for s, _ in sorted(
            zip(sentences, similarities),
            key=lambda x: x[1],
            reverse=True,
        )
    ]


with open(SRC, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

changed = 0
top1_changed = 0

with open(DST, "w", encoding="utf-8") as fout:

    for x in tqdm(data):
        old = x["sentences"]
        new = sorted_sentences(
            x["question"],
            old
        )

        if old != new:
            changed += 1

        if old and new and old[0] != new[0]:
            top1_changed += 1

        # 只改变 sentences 的顺序
        x["sentences"] = new

        fout.write(
            json.dumps(x, ensure_ascii=False) + "\n"
        )

print("total:", len(data))
print("order changed:", changed)
print("changed ratio:", changed / len(data))
print("top1 changed:", top1_changed)
print("top1 changed ratio:", top1_changed / len(data))
print("saved:", DST)
