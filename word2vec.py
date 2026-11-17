import re
from collections import Counter

import numpy as np
from nltk.tokenize import sent_tokenize
from numpy.typing import NDArray


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    # Wrap NLTK's sentence tokenizer so I can experiment with a custom implementation later
    sentences = sent_tokenize(text)
    return sentences


def tokenize(text: str) -> list[str]:
    """Convert a string into a list of lowercase words."""
    text = text.lower()
    text = text.replace('’', "'")  # Normalize curly apostrophes
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)*", text)
    return tokens


def build_vocab(
    tokenized_sentences: list[list[str]], min_count: int = 1
) -> tuple[dict[str, int], dict[int, str], list[int]]:
    """Build word/ID lookup dicts and list of word counts aligned to ID's."""
    initial_counts: Counter[str] = Counter()

    for sentence in tokenized_sentences:
        initial_counts.update(sentence)

    vocab = sorted(word for word, count in initial_counts.items() if count >= min_count)

    id_to_word: dict[int, str] = {}
    word_to_id: dict[str, int] = {}
    word_counts: list[int] = []

    for word_id, word in enumerate(vocab):
        id_to_word[word_id] = word
        word_to_id[word] = word_id
        word_counts.append(initial_counts[word])

    return word_to_id, id_to_word, word_counts


def generate_pairs(
    tokenized_sentences: list[list[str]],
    word_to_id: dict[str, int],
    window_size: int = 2,
) -> list[tuple[int, int]]:
    """Create (center_word_id, context_word_id) pairs."""
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


def vocab_coverage(
    tokenized_sentences: list[list[str]], word_to_id: dict[str, int]
) -> tuple[float, float]:
    """Return (token coverage, type coverage) as fractions."""
    tokens = []

    for sentence in tokenized_sentences:
        tokens.extend(sentence)

    known_tokens = [word in word_to_id for word in tokens]
    token_coverage = sum(known_tokens) / len(tokens)

    unique_words = set(tokens)
    known_types = [word in word_to_id for word in unique_words]
    type_coverage = sum(known_types) / len(unique_words)

    return token_coverage, type_coverage


class Word2Vec:
    """Skip-gram model implemented with NumPy."""

    def __init__(
        self,
        word_to_id: dict[str, int],
        id_to_word: dict[int, str],
        embedding_dim: int,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.word_to_id = word_to_id.copy()
        self.id_to_word = id_to_word.copy()
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        self.vocab_size = len(self.word_to_id)
        self.input_embeddings, self.output_embeddings = self._initialize_embeddings()
        self.loss_history: list[float] = []

    def _initialize_embeddings(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create input and output embedding matrices with random values."""
        input_embeddings = self.rng.normal(
            loc=0.0, scale=0.1, size=(self.vocab_size, self.embedding_dim)
        )
        output_embeddings = self.rng.normal(
            loc=0.0, scale=0.1, size=(self.vocab_size, self.embedding_dim)
        )

        return input_embeddings, output_embeddings

    def calculate_scores(self, center_id: int) -> NDArray[np.float64]:
        """Calculate context-prediction score for every vocabulary word."""
        center_vector = self.input_embeddings[center_id]
        scores = self.output_embeddings @ center_vector

        return scores

    @staticmethod
    def softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert a 1d array of scores into probabilities."""
        shifted = scores - np.max(scores)
        weights = np.exp(shifted)
        probabilities = weights / np.sum(weights)

        return probabilities

    @staticmethod
    def cross_entropy(scores: NDArray[np.float64], context_id: int) -> float:
        """Calculates cross-entropy loss from scores and the target context ID."""
        # Calculate using scores to avoid ln(0) = undefined
        shifted = scores - np.max(scores)
        weights = np.exp(shifted)
        loss = np.log(np.sum(weights)) - shifted[context_id]

        return float(loss)

    def backward(
        self,
        probabilities: NDArray[np.float64],
        context_id: int,
        center_vector: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return gradients for the center vector and output embedding matrix."""
        score_gradients = probabilities.copy()
        score_gradients[context_id] -= 1.0

        output_gradients = np.outer(score_gradients, center_vector)
        center_gradient = self.output_embeddings.T @ score_gradients

        return center_gradient, output_gradients

    def evaluate(self, pairs: list[tuple[int, int]]) -> float:
        """Return average cross-entropy for given pairs."""
        total_loss = 0.0

        for center_id, context_id in pairs:
            scores = self.calculate_scores(center_id)
            total_loss += self.cross_entropy(scores, context_id)

        return total_loss / len(pairs)

    def train(
        self, train_pairs: list[tuple[int, int]], epochs: int = 100
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float]]:
        """Train embeddings and return both matrices and epoch losses."""
        for epoch in range(epochs):
            total_loss = 0.0

            # Shuffle each epoch to help converge more reliably
            # Helps minimize effects from repeatedly updating on pairs in the same order
            pair_order = self.rng.permutation(len(train_pairs))

            for i in pair_order:
                center_id, context_id = train_pairs[i]

                # Forward pass
                center_vector = self.input_embeddings[center_id]
                scores = self.calculate_scores(center_id)
                probabilities = self.softmax(scores)

                # Record loss
                loss = self.cross_entropy(scores, context_id)
                total_loss += loss

                # Calculate gradients
                center_gradient, output_gradients = self.backward(
                    probabilities, context_id, center_vector
                )

                # Gradient descent update
                self.input_embeddings[center_id] -= self.learning_rate * center_gradient
                self.output_embeddings -= self.learning_rate * output_gradients

            avg_loss = total_loss / len(train_pairs)
            self.loss_history.append(avg_loss)

        return self.input_embeddings, self.output_embeddings, self.loss_history
