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


def random_split(
    tokenized_sentences: list[list[str]], split_at: tuple[float, ...], seed: int = 42
) -> list[list[list[str]]]:
    """Shuffle tokenized sentences and split them at given fractions."""
    size = len(tokenized_sentences)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(size)
    shuffled_sentences = [tokenized_sentences[i] for i in indices]

    splits: list[list[list[str]]] = []
    start = 0

    for frac in split_at:
        end = int(frac * size)
        splits.append(shuffled_sentences[start:end])
        start = end

    splits.append(shuffled_sentences[start:])
    return splits


def build_vocab(
    tokenized_sentences: list[list[str]], min_count: int = 1
) -> tuple[dict[str, int], dict[int, str], list[int]]:
    """Build word/ID lookup dicts and list of word counts aligned to IDs."""
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


def vocab_coverage(
    tokenized_sentences: list[list[str]], word_to_id: dict[str, int]
) -> tuple[float, float]:
    """Return (token coverage, type coverage) as fractions."""
    tokens = []

    for sentence in tokenized_sentences:
        tokens.extend(sentence)

    known_token_count = [word in word_to_id for word in tokens]
    token_coverage = sum(known_token_count) / len(tokens)

    unique_words = set(tokens)
    known_type_count = [word in word_to_id for word in unique_words]
    type_coverage = sum(known_type_count) / len(unique_words)

    return token_coverage, type_coverage


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


def pca_2d(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return 2d PCA coordinates for each word vector."""
    centered = vectors - vectors.mean(axis=0)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    axes = directions[:2].T
    return (centered @ axes).astype(np.float32)


class Word2Vec:
    """Skip-gram model implemented with NumPy."""

    def __init__(
        self,
        word_to_id: dict[str, int],
        id_to_word: dict[int, str],
        embedding_dim: int,
        seed: int = 42,
    ) -> None:
        self.word_to_id = word_to_id.copy()
        self.id_to_word = id_to_word.copy()
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)

        self.vocab_size = len(self.word_to_id)
        self.input_embeddings, self.output_embeddings = self._initialize_embeddings()

    def train(
        self,
        train_pairs: list[tuple[int, int]],
        val_pairs: list[tuple[int, int]],
        epochs: int,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        restore_best_weights: bool = False,
    ) -> tuple[list[float], list[float]]:
        """Train embeddings and return train and validation loss histories.

        Training loss is averaged during each epoch as weights are updated.
        Validation loss is computed using the weights at the end of each epoch.
        """
        train_pairs_array = np.asarray(train_pairs, dtype=np.intp)

        train_loss_history: list[float] = []
        val_loss_history: list[float] = []

        best_val_loss = float('inf')
        best_embeddings = None

        for _ in range(epochs):
            train_loss = self._train_epoch(train_pairs_array, batch_size, learning_rate)
            val_loss = self.evaluate(val_pairs, batch_size)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            # Save weights with lowest val loss if desired
            if restore_best_weights and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_embeddings = (
                    self.input_embeddings.copy(),
                    self.output_embeddings.copy(),
                )

        # Restore best epoch's weights if desired
        if restore_best_weights and best_embeddings is not None:
            self.input_embeddings, self.output_embeddings = best_embeddings

        return train_loss_history, val_loss_history

    def evaluate(self, pairs: list[tuple[int, int]], batch_size: int = 128) -> float:
        """Return average cross-entropy for the given pairs."""
        pairs_array = np.asarray(pairs, dtype=np.intp)
        split_points = range(batch_size, len(pairs_array), batch_size)
        batches = np.split(pairs_array, split_points)

        total_loss = 0.0
        for batch in batches:
            center_ids, context_ids = batch.T
            center_vectors = self.input_embeddings[center_ids]
            scores = self._context_scores(center_vectors)
            losses = self._cross_entropy(scores, context_ids)
            total_loss += float(losses.sum(dtype=np.float64))

        return total_loss / len(pairs_array)

    def cosine_similarity(self, word_a: str, word_b: str) -> float:
        """Return cosine similarity between two words' input embeddings."""
        vec_a = self.input_embeddings[self.word_to_id[word_a]]
        vec_b = self.input_embeddings[self.word_to_id[word_b]]

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        return float((vec_a @ vec_b) / (norm_a * norm_b))

    def most_similar(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        """Return k (word, similarity) pairs, highest similarity first."""
        neighbors = []
        for candidate in self.word_to_id:
            if candidate == word:
                continue

            score = self.cosine_similarity(word, candidate)
            neighbors.append((candidate, score))

        neighbors.sort(key=lambda pair: pair[1], reverse=True)
        return neighbors[:k]

    def predict_context(self, center_word: str, k: int = 5) -> list[tuple[str, float]]:
        """Return k (context word, probability) pairs, highest probability first."""
        center_id = self.word_to_id[center_word]
        scores = self._context_scores(self.input_embeddings[center_id])
        probabilities = self._softmax(scores)
        top_ids = np.argsort(-scores)[:k]

        predictions = []
        for word_id in top_ids:
            word = self.id_to_word[word_id]
            probability = float(probabilities[word_id])
            predictions.append((word, probability))
        return predictions

    def _train_epoch(
        self, pair_ids: NDArray[np.intp], batch_size: int, learning_rate: float
    ) -> float:
        """Shuffle and train one pass and return mean loss."""
        # Shuffle pairs before batching to vary their order and grouping each epoch
        shuffled_pairs = self.rng.permutation(pair_ids, axis=0)

        split_points = range(batch_size, len(shuffled_pairs), batch_size)
        batches = np.split(shuffled_pairs, split_points)

        total_loss = 0.0
        for batch in batches:
            center_ids, context_ids = batch.T

            # Forward pass
            center_vectors = self.input_embeddings[center_ids]

            scores = self._context_scores(center_vectors)
            probabilities = self._softmax(scores)

            # Record loss
            losses = self._cross_entropy(scores, context_ids)
            total_loss += float(losses.sum(dtype=np.float64))

            # Calculate gradients
            center_gradients, output_gradients = self._backward(
                probabilities, context_ids, center_vectors
            )

            # Gradient descent update
            # Accumulate all updates when center ID repeats
            np.add.at(
                self.input_embeddings,
                center_ids,
                -learning_rate * center_gradients,
            )
            self.output_embeddings -= learning_rate * output_gradients

        return total_loss / len(pair_ids)

    def _context_scores(
        self, center_vectors: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Calculate context-prediction scores for one center vector or a batch."""
        return center_vectors @ self.output_embeddings.T

    @staticmethod
    def _softmax(scores: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert scores to probabilities along vocab axis."""
        shifted = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(shifted)
        probabilities = weights / weights.sum(axis=-1, keepdims=True)
        return probabilities.astype(np.float32, copy=False)

    @staticmethod
    def _cross_entropy(
        scores: NDArray[np.float32], context_ids: NDArray[np.intp]
    ) -> NDArray[np.float32]:
        """Calculate cross-entropy loss for each pair in a batch."""
        shifted = scores - scores.max(axis=1, keepdims=True)
        log_normalizers = np.log(np.exp(shifted).sum(axis=1))
        rows = np.arange(len(context_ids))
        losses = log_normalizers - shifted[rows, context_ids]
        return losses

    def _backward(
        self,
        probabilities: NDArray[np.float32],
        context_ids: NDArray[np.intp],
        center_vectors: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return gradients for the mean batch loss."""
        score_gradients = probabilities.copy()
        rows = np.arange(len(context_ids))
        score_gradients[rows, context_ids] -= 1.0
        score_gradients /= len(context_ids)

        center_gradients = score_gradients @ self.output_embeddings
        output_gradients = score_gradients.T @ center_vectors
        return center_gradients, output_gradients

    def _initialize_embeddings(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Create input and output embedding matrices with random values."""
        input_embeddings = self.rng.normal(
            loc=0.0, scale=0.1, size=(self.vocab_size, self.embedding_dim)
        ).astype(np.float32)
        output_embeddings = self.rng.normal(
            loc=0.0, scale=0.1, size=(self.vocab_size, self.embedding_dim)
        ).astype(np.float32)
        return input_embeddings, output_embeddings
