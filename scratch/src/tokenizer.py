"""SentencePiece unigram tokeniser, trained per language on its full
training split. Wrapped as a HuggingFace fast tokeniser for use with the
standard ``Trainer``/``DataCollatorWithPadding`` flow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers.implementations import SentencePieceUnigramTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CLS_TOKEN = "<cls>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN]


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    out_path: Path,
) -> None:
    """Train a SentencePiece unigram tokeniser and save it to ``out_path``.

    The post-processor prepends a ``<cls>`` token to every input so the model
    can use position 0 as the pooled representation.
    """
    tok = SentencePieceUnigramTokenizer()
    tok.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        unk_token=UNK_TOKEN,
    )
    cls_id = tok.token_to_id(CLS_TOKEN)
    tok.post_processor = TemplateProcessing(
        single=f"{CLS_TOKEN} $A",
        special_tokens=[(CLS_TOKEN, cls_id)],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))


def load_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    """Load a saved tokeniser as a HuggingFace fast tokeniser."""
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        cls_token=CLS_TOKEN,
    )
