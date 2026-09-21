import re

from nltk.tokenize import sent_tokenize
from collections import Counter


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    sentences = sent_tokenize(text)
    return sentences


def tokenize(text: str) -> list[str]:
    """Convert a string into a list of lowercase words."""
    text = text.lower()
    # Replace curly apostrophes with straight ones
    text = text.replace("’", "'")
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)*", text)
    return tokens


def build_vocab(tokenized_sentences: list[list[str]], min_count: int = 1) -> tuple[dict[int, str], dict[str, int], list[int]]:
    """Build word/ID lookup dicts and list of word counts aligned to ID's."""
    if min_count < 1:
        raise ValueError("min_count must be at least 1")

    initial_counts: Counter[str] = Counter()

    for sentence in tokenized_sentences:
        initial_counts.update(sentence)

    vocab = sorted(
        word for word, count in initial_counts.items()
        if count >= min_count
    )

    id_to_word: dict[int, str] = {}
    word_to_id: dict[str, int] = {}
    word_counts: list[int] = []

    for word_id, word in enumerate(vocab):
        id_to_word[word_id] = word
        word_to_id[word] = word_id
        word_counts.append(initial_counts[word])

    return id_to_word, word_to_id, word_counts

def generate_pairs(tokenized_sentences: list[list[str]], word_to_id: dict[str, int], window_size: int = 2) -> list[tuple[int, int]]:
    """Create (center_word_id, context_word_id) pairs."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    pairs: list[tuple[int, int]] = []

    for sentence in tokenized_sentences:
        for center_pos, center_word in enumerate(sentence):
            center_id = word_to_id.get(center_word)

            if center_id is None:
                continue

            start = max(0, center_pos - window_size)
            stop = min(len(sentence), center_pos + window_size + 1)

            for context_pos in range(start, stop):
                if context_pos == center_pos:
                    continue

                context_word = sentence[context_pos]
                context_id = word_to_id.get(context_word)

                if context_id is None:
                    continue

                pairs.append((center_id, context_id))

    return pairs