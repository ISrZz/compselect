import json
import argparse


# Faithful implementation of the Adaptive Truncator prompt in Appendix A.2.
SYSTEM_PROMPT = (
    "You are a highly skilled assistant for improving language model efficiency "
    "through context truncation. Given a question and a ranked list of sentences, "
    "retain the sentences that are most relevant and remove irrelevant context."
)


def format_sentences(sentences):
    return "\n".join(
        f"Sentence {i}: {sentence}"
        for i, sentence in enumerate(sentences, start=1)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []

    empty_final = 0

    for item in data:
        question = item["question"]
        summary = item["summary"]
        final = item["final"]

        # Standard truncator annotation:
        # samples for which no sufficient prefix was found are not
        # usable as ordinary positive SFT targets.
        if not final:
            empty_final += 1
            continue

        ranked_list = format_sentences(summary)
        target = format_sentences(final)

        output.append(
            {
                "instruction": SYSTEM_PROMPT,
                "input": (
                    f"Question:\n{question}\n\n"
                    f"Ranked List:\n{ranked_list}"
                ),
                "output": target,
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Original samples:", len(data))
    print("Removed empty-final samples:", empty_final)
    print("SFT samples:", len(output))
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()
