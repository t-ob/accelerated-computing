---
title: "Notes on neural networks: positional encoding"
date: 2023-12-07T22:57:57+01:00
draft: true
math: true
---

Disclaimer: this post contains content which might be just simply incorrect. I'm still internalising some of these concepts, and my hope is the act of writing them down will help to make them a bit more concrete. This is very much a "thinking out loud" kind of post.

### Context windows and positions

I will write about attention and the transformer architecture in a (/ a number of) future posts, but their study has motivated this one so I'll need to use a bit of that language to set the scene.

A transformer network takes a "context window" as its input. A context window is a sequence of length $n\_{\text{context}}$ of (row) vectors in some embedding space of dimension $n\_{\text{embed}}$. For example, if we have a two-dimensional word embedding, we might represent the sentence "the dog is good" as the matrix

$$
\begin{pmatrix}
   0.1 & -0.3 \\\\
   0.6 & 0.2 \\\\
   -0.4 & -0.1 \\\\
   0.2 & -0.7
\end{pmatrix}
$$

where eg. the embedding of "dog" is $(0.6, 0.2) \in \mathbb{R}^2$. In PyTorch, this might be represented by the following tensor of shape `(4, 2)`:

```python
the_dog_is_good = torch.tensor(
    [[0.1, -0.3], [0.6, 0.2], [-0.4, -0.1], [0.2, -0.7]]
)
```

There is a point during the calculation of attention scores where information of the positions of elements within their context is lost, so that eg. (in the absence of some additional processing) the sentences "only who can prevent forest fires" and "who can prevent forest fires only" would appear as indistinguishable to the network, despite having different meanings.

### The trick

One solution to this problem is to use positional encoding. My high level intuition on this is as follows:

1. We start with some vocabulary of size $n\_{\text{vocab}}$ (eg. words in the english dictionary, [BPE-generated][bpe] subwords of scraped internet content, etc.).
2. Each item in this vocabulary gets embedded into some $n\_{\text{embed}}$ dimensional space.
3. To account for position within a given context, a new composite embedding is formed by taking the original embedding, and creating $n\_{\text{context}}$ new values for each vector in its image. Each value is translated by $n\_{\text{context}}$ vectors.
4. The network then takes its inputs from values this new vocabulary, which has size $n\_{\text{vocab}} \cdot n\_{\text{context}}$.

So, continuing our example of of embeddings in space of dimension $n\_{\text{embed}} = 2$, with context windows of length $n\_{\text{context}} = 4$, we would like four positional translations $p_i = (p\_{i, x}, p\_{i, y})$ for $i = 1, \ldots, 4$, so that any $(x, y)$ occurring at position $i$ becomes $(x + p\_{i, x}, y + p\_{i, y})$ by the time the network sees it.

### Constructing the embeddings

In the paper [Attention Is All You Need][paper-aiayn], the authors present two concrete approaches -- one with no additional network parameters and a second, simpler, version, which comes at the cost of an additional embedding matrix to learn.

#### Sinusoidal method

The idea is to translate each point in the embedding space by one of $n\_{\text{context}}$ points $p_i$, whose $k$-th component $p_{i, k}$ is defined to be

$$
p_{i, k} = \begin{cases} 
\sin(\frac{i}{10000^{\frac{k}{n\_{\text{embed}}}}}) &\text{if } k \text{ is even,} \\\\
\cos(\frac{i}{10000^{\frac{k - 1}{n\_{\text{embed}}}}}) &\text{if } k \text{ is odd.}
\end{cases}
$$

Note that by considering the squares of each even-odd pair $(p\_{i,k}, p\_{i,k+1})$, we see that each $p_i$ lies on the sphere centered at the origin with radius $\sqrt{\frac{n\_{\text{embed}}}{2}}$ -- the original vocabulary is translated (by equal amounts across all possible positions) in different directions determined by each $p_i$. [^my-ignorance]

We can play around with the kinds of positional encodings for various context lengths and embedding dimensions with the following script:

```python
import torch

CONTEXT_LENGTH = 4
EMBEDDING_DIM = 8


def make_positional_embedding(context_length, embedding_dim):
    positions = torch.arange(context_length).float()
    coefficients = 10000 ** -(
        ((torch.arange(embedding_dim).int() >> 1) << 1) / embedding_dim
    )
    radians = positions.view(-1, 1) @ coefficients.view(1, -1)
    print(f"{radians=}")

    evens = torch.arange(0, embedding_dim, step=2)
    odds = torch.arange(1, embedding_dim, step=2)

    encodings = torch.zeros_like(radians)
    encodings[:, evens] = torch.sin(radians[:, evens])
    encodings[:, odds] = torch.cos(radians[:, odds])

    return encodings


if __name__ == "__main__":
    positional_embedding = make_positional_embedding(CONTEXT_LENGTH, EMBEDDING_DIM)

    print(f"{positional_embedding=}")
    print(f"{positional_embedding.norm(dim=1)=}")
```

which for the context length and embedding dims as written produces the following output:

```
radians=tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
         0.0000e+00, 0.0000e+00],
        [1.0000e+00, 1.0000e+00, 1.0000e-01, 1.0000e-01, 1.0000e-02, 1.0000e-02,
         1.0000e-03, 1.0000e-03],
        [2.0000e+00, 2.0000e+00, 2.0000e-01, 2.0000e-01, 2.0000e-02, 2.0000e-02,
         2.0000e-03, 2.0000e-03],
        [3.0000e+00, 3.0000e+00, 3.0000e-01, 3.0000e-01, 3.0000e-02, 3.0000e-02,
         3.0000e-03, 3.0000e-03]])
encodings=tensor([[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
          1.0000e+00,  0.0000e+00,  1.0000e+00],
        [ 8.4147e-01,  5.4030e-01,  9.9833e-02,  9.9500e-01,  9.9998e-03,
          9.9995e-01,  1.0000e-03,  1.0000e+00],
        [ 9.0930e-01, -4.1615e-01,  1.9867e-01,  9.8007e-01,  1.9999e-02,
          9.9980e-01,  2.0000e-03,  1.0000e+00],
        [ 1.4112e-01, -9.8999e-01,  2.9552e-01,  9.5534e-01,  2.9995e-02,
          9.9955e-01,  3.0000e-03,  1.0000e+00]])
encodings.norm(dim=1)=tensor([2., 2., 2., 2.])
```

Putting this together in a PyTorch module might look like:

```python
VOCAB_SIZE = 128
EMBEDDING_DIM = 64
CONTEXT_LENGTH = 8


class NetWithSinusoidalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.register_buffer(
            "pos", make_positional_embedding(CONTEXT_LENGTH, EMBEDDING_DIM)
        )
        # ... other layers

    def forward(self, x):
        x = self.emb(x)
        x = x + self.pos
        # ... rest of forward pass
```

#### Just let the network learn it

When I first read the above paper, this alternative approach -- to just learn an embedding -- seemed preferable to me, but I appreciate now the touch of class the sinusoidal approach brings to the table.

The idea is to equip the network with an additional embedding and let it figure out how to use it to distinguish between positions. It's less code, at the cost of some additional parameters to train. In PyTorch, it might look like this:

```python
import torch
import torch.nn as nn

VOCAB_SIZE = 128
EMBEDDING_DIM = 64
CONTEXT_LENGTH = 8


class NetWithLearnedPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.pos = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_DIM)
        # ... other layers

    def forward(self, x):
        x = self.emb(x)
        p = self.pos(torch.arange(CONTEXT_LENGTH))
        x = x + p
        # ... rest of forward pass
```

The paper claims that in practice, both approaches yielded more or less identical results, so both approaches appear to be about as effective as each other.

### Alternate approaches

Other approaches to positional encodings exist and have been explored since the transformer architecture exploded in popularity. My fellow RC participant [Régis][regis-blog] ran a few sessions to explore these further. We looked at [RoPE][paper-rope], which replaces positional encoding with a sequence of rotations, and [ALiBi][paper-alibi], which substitutes positional encoding altogether with a modified query-key attention score process, which penalises attention scores between items that are far apart.

### Conclusion

Putting these thoughts into words has made me realise there's a good chance I'm missing some subtleties around positional embeddings, and indeed embeddings in general. An old colleague of mine has recommended [Embeddings in Natural Language Processing][book-nlp] -- maybe now's a good time to pick it up.

### Further reading

- [Attention Is All You Need][paper-aiayn]
- [RoFormer: Enhanced Transformer with Rotary Position Embedding][paper-rope]
- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation][paper-alibi]
- [Embeddings in Natural Language Processing: Theory and Advances in Vector Representations of Meaning][book-nlp]

[paper-aiayn]: https://arxiv.org/abs/1706.03762
[paper-rope]: https://arxiv.org/abs/2104.09864
[paper-alibi]: https://arxiv.org/abs/2108.12409
[book-nlp]: https://sites.google.com/view/embeddings-in-nlp
[bpe]: https://en.wikipedia.org/wiki/Byte_pair_encoding
[regis-blog]: https://swe-to-mle.pages.dev/

[^my-ignorance]: I still find this surprising as the intuition I've internalised, which I now believe is not correct, is that any respectable positional embedding would not perturb the original vocabulary too much, so as to preserve semantic meanings (in the case of natural language, at least). In this case as our embedding dimension grows we end up pushing our points in different positions further and further apart. Maybe semantic meaning (eg. the classic example "king - man + woman = queen") has less to do with absolute position and more to do with relative position amongst neighbouring points.