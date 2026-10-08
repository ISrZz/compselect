import argparse
import json
import re

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


INSTRUCTION = (
    "You are a highly skilled assistant for improving language model efficiency "
    "through context truncation. Given a question and a ranked list of sentences, "
    "retain the sentences that are most relevant and remove irrelevant context."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def clean_summary(summary):
    if isinstance(summary, list):
        return summary

    result = []
    for line in summary.splitlines():
        line = line.strip()
        if not line:
            continue

        line = re.sub(r"^Sentence\s+\d+\s*:\s*", "", line)
        result.append(line)

    return result


def parse_generated(text):
    result = []

    for line in text.splitlines():
        line = line.strip()

        m = re.match(r"^Sentence\s+\d+\s*:\s*(.*)$", line)
        if m and m.group(1).strip():
            result.append(m.group(1).strip())

    return result


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    with open(args.output, "w", encoding="utf-8") as fout:

        for x in tqdm(data):

            summary = clean_summary(x["summary"])

            ranked_list = "\n".join(
                f"Sentence {i}: {sentence}"
                for i, sentence in enumerate(summary, 1)
            )

            input_text = (
                f"Question:\n{x['question']}\n\n"
                f"Ranked List:\n{ranked_list}"
            )

            # 对应训练数据的 instruction + input
            user_content = INSTRUCTION + "\n\n" + input_text

            messages = [
                {"role": "user", "content": user_content}
            ]

            tokenized = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )

            if hasattr(tokenized, "input_ids"):
                input_ids = tokenized.input_ids.to(model.device)
                attention_mask = getattr(tokenized, "attention_mask", None)

                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                else:
                    attention_mask = attention_mask.to(model.device)

            elif torch.is_tensor(tokenized):
                input_ids = tokenized.to(model.device)
                attention_mask = torch.ones_like(input_ids)

            else:
                input_ids = tokenized["input_ids"].to(model.device)
                attention_mask = tokenized.get("attention_mask")

                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                else:
                    attention_mask = attention_mask.to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(
                output_ids[0, input_ids.shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            selected = parse_generated(generated)

            # 当前目标只是先把 pipeline 跑通。
            # 格式异常时保底使用 ranked clue 第一条。
            if not selected and summary:
                selected = summary[:1]

            result = {
                "question": x["question"],
                "answer": x["answer"],

                # Generator 的 inference_llama.py 读取这个字段
                "documents": selected,

                # 留下来方便检查
                "summary": summary,
                "truncator_output": generated,
            }

            fout.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )


if __name__ == "__main__":
    main()
