from app.rag.tokenization import count_tokens


class FakeTokenizer:
    model_max_length = 512

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> list[int]:
        tokens = text.split()

        token_ids = list(range(len(tokens)))

        if add_special_tokens:
            return [-1, *token_ids, -2]

        return token_ids


def test_count_tokens_includes_special_tokens() -> None:
    tokenizer = FakeTokenizer()

    assert count_tokens(
        "Medicare Part B coverage",
        tokenizer,
    ) == 6


def test_count_tokens_does_not_require_truncation() -> None:
    tokenizer = FakeTokenizer()

    long_text = "word " * 600

    assert count_tokens(
        long_text,
        tokenizer,
    ) == 602