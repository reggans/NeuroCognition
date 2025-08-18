from __future__ import annotations
from .api import generate_text_rapm_item


def smoke(n: int = 5):
    for seed in range(1, n + 1):
        try:
            item = generate_text_rapm_item(seed=seed, debug=True)
            print(
                f"Seed {seed} OK; answer={item['answer']} options={len(item['options'])}"
            )
        except Exception as e:
            print(f"Seed {seed} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    smoke(20)
