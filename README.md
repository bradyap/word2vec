# Word2Vec

A NumPy implementation of the Word2Vec skip-gram model. The model learns word embeddings by predicting surrounding words from a target word, giving words that appear in similar contexts similar vectors.

The demo notebook trains the model on a corpus of science articles, compares it with a context-frequency baseline, and visualizes the learned word embeddings.

[Model implementation](word2vec.py) · [Demo notebook](demo.ipynb) · [Presentation (PDF)](docs/presentation.pdf)

## Run the notebook

```bash
uv sync
uv run jupyter notebook demo.ipynb
```

## Reference

Mikolov et al. (2013), [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781).
