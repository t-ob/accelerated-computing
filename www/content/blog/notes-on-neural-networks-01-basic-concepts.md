---
title: "Notes on neural networks: basic concepts"
date: 2023-10-02T14:25:42+01:00
draft: false
math: true
categories:
- nn-notes
---

### Overview

A neural network is (broadly speaking) a collection of transformations arranged in a sequence of layers of neurons. A neuron is made up of some weights $W$, a bias $b$, and (frequently) a non-linear function $g$. Examples of non-linear functions typically used are ReLU, tanh, softmax, and sigmoid. A neuron in layer $i$ takes as inputs the outputs of the neurons in layer $i - 1$, applies the linear transformation encoded by its weights, adds its bias, then finally outputs the value of $g$ applied to the sum. The neurons, together with their inputs and outputs, form a computation graph, where an edge from neuron $s$ to neuron $t$ indicates that the output of $s$ is used in the computation of $t$. In practice, networks are often created with random weights and biases equal to zero.

Neural networks are trained by minimising a loss function $L$. Given some training examples $X$ and labels $Y$, if $\hat{Y}$ denotes the output of a network on $X$, then $L = L(\hat{Y}, Y)$ should be chosen such that it takes non-negative values, and such that the closer $\hat{Y}$ is to $Y$, the smaller $L$ becomes. Examples of loss functions include mean squared error and binary cross entropy. Minimising the loss is achieved through adjusting all the weights and biases in the network in a way that minimises the distance between $\hat{Y}$ and $Y$. This is done via the process of [backpropagation][backprop] -- iterating over the computation graph backwards from $L$, computing the derivatives of all weights and biases with respect to $L$, and using those to update the network parameters (the set of all weights and biases) in such a way that results in a reduced loss. For example, during gradient descent, a network parameter is updated by subtracting from it some small multiple -- known as the learning rate -- of its gradient.

Backpropagation is a topic worthy of its own post, but the general idea is that by successively applying the chain rule to nodes in our computation graph -- in a carefully chosen order -- we can compute the gradient of our network parameters with respect to some loss, no matter how deep the network architecture.

### Example

Let's use [PyTorch][pytorch][^pytorch-disclaimer] to try to learn the function which takes a point on the plane and maps it to $1$ if it lies within the unit circle, and zero otherwise:

$$
(x_1, x_2) \mapsto \begin{cases}
   1 &\text{if } x_1^2 + x_2^2 < 1 \\\\
   0 &\text{otherwise}
\end{cases}
$$

We'll construct a network with three layers: one hidden layer with ten neurons, one hidden layer with five neurons, and a final output layer with a single neuron. For the non-linear functions, we'll use ReLU in the hidden layers, and sigmoid for the output layer.[^architecture] We create a synthetic dataset of $10,000$ points, partitioned (in a roughly $80$:$20$ split) into $n_{\text{train}}$ training examples, and $n_{\text{dev}}$ examples to be held back, only to be used in evaluating our model's performance after training has complete. To keep things simple, we'll use a fixed epoch count (the number of iterations of the training loop performed) of $1,000$ and learning rate of $0.5$.

Before we dive into the code, let's have a look at the dimensions of our inputs, outputs, and network parameters:

| Name | Dimension | Description |
|---|-|---|
|X_train  | ($n_{\text{train}}$, $2$) | Inputs to network, arranged in $n_{\text{train}}$ rows of coordinate pairs |
|Y_train  | ($n_{\text{train}}$, $1$) | Training labels of ones and zeros, arranged in $n_{\text{train}}$ rows |
|X_dev  | ($n_{\text{dev}}$, $2$) | Input examples held back from training, arranged in $n_{\text{dev}}$ rows |
|Y_dev  | ($n_{\text{dev}}$, $1$) | Training labels held back from training, arranged in $n_{\text{train}}$ rows |
|W1 | ($2$, $10$) | Weights for the ten neurons in the first layer [^dimensions] |
|b1 | ($10$) | Biases for the ten neurons in the first layer [^broadcasting] |
|W2 | ($10$, $5$) | Weights for the five neurons in the second layer |
|b2 | ($5$) | Biases for the five neurons in the second layer |
|W3 | ($5$, $1$) | Weights for the single neuron in the output layer |
|b3 | ($1$) | Bias for the single neurons in the output layer |

We begin by creating our synthetic dataset (and acknowledging how contrived this example is -- we're using the very function that we're trying to learn to generate them!) and initialising the parameters of our network. We then train the model, by performing the following steps in a loop:

1. Perform a forward pass of our inputs through the network.
2. Compute the training loss at the current iteration.
3. Backpropagate through that loss.
4. Update each param by subtracting a small fraction of its gradient (with respect to the loss), and then resetting its gradient for the next iteration.

After training is complete, we evaluate our model against our held back data, and print out some statistics, as well as some examples of false positive and negative classifications.

The complete code listing follows:

```python
import typing

import torch


Dataset = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

NeuralNetwork = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def generate_datasets(generator: torch.Generator) -> Dataset:
    X = torch.rand(10000, 2, generator=generator) * 10.0 - 5.0
    Y = ((X**2).sum(dim=1, keepdim=True) < 1.0).float()

    P = torch.rand(10000, generator=generator)

    X_train = X[P < 0.8]
    Y_train = Y[P < 0.8]

    X_dev = X[P >= 0.8]
    Y_dev = Y[P >= 0.8]

    return X_train, Y_train, X_dev, Y_dev


def init_network(generator) -> NeuralNetwork:
    W1 = torch.randn(2, 10, generator=generator, requires_grad=True)
    b1 = torch.zeros(10, requires_grad=True)

    W2 = torch.randn(10, 5, generator=generator, requires_grad=True)
    b2 = torch.zeros(5, requires_grad=True)

    W3 = torch.randn(5, 1, generator=generator, requires_grad=True)
    b3 = torch.zeros(1, requires_grad=True)

    return W1, b1, W2, b2, W3, b3


def forward_pass(params: NeuralNetwork, X: torch.Tensor) -> torch.Tensor:
    W1, b1, W2, b2, W3, b3 = params
    Z1 = (X @ W1 + b1).relu()
    Z2 = (Z1 @ W2 + b2).relu()
    Z3 = (Z2 @ W3 + b3).sigmoid()

    return Z3


def mean_squared_error(A, Y):
    return ((A - Y) ** 2).squeeze().mean()


def fit(params: NeuralNetwork, X: torch.Tensor, Y: torch.Tensor):
    for _ in range(1000):
        Z = forward_pass(params, X)

        L = mean_squared_error(Z, Y)

        L.backward()

        for param in params:
            param.data -= 0.5 * typing.cast(torch.Tensor, param.grad)
            param.grad = None


if __name__ == "__main__":
    generator = torch.Generator()
    generator.manual_seed(1337)

    X_train, Y_train, X_dev, Y_dev = generate_datasets(generator)

    params = init_network(generator)

    fit(params, X_train, Y_train)

    with torch.no_grad():
        Y_dev_hat = forward_pass(params, X_dev)
        dev_loss = mean_squared_error(Y_dev_hat, Y_dev)
        Y_dev_hat = (Y_dev_hat > 0.5).float()

    tp = (Y_dev == 1.0) & (Y_dev_hat == 1.0)
    fp = (Y_dev == 0.0) & (Y_dev_hat == 1.0)
    tn = (Y_dev == 0.0) & (Y_dev_hat == 0.0)
    fn = (Y_dev == 1.0) & (Y_dev_hat == 0.0)

    true_positives = tp.float().sum().item()
    false_positives = fp.float().sum().item()
    true_negatives = tn.float().sum().item()
    false_negatives = fn.float().sum().item()

    false_positive_samples = X_dev[fp.squeeze(), :]
    false_negative_samples = X_dev[fn.squeeze(), :]

    print(f"{dev_loss=}")
    print(f"{true_positives=}")
    print(f"{false_positives=}")
    print(f"{true_negatives=}")
    print(f"{false_negatives=}")
    print(f"{false_positive_samples=}")
    print(f"{false_negative_samples=}")
```

When running, we see the following output:

```
dev_loss=tensor(0.0027)
true_positives=45.0
false_positives=1.0
true_negatives=1965.0
false_negatives=4.0
false_positive_samples=tensor([[-0.5229, -0.9005]])
false_negative_samples=tensor([[-0.8656,  0.3219],
        [ 0.8683,  0.4416],
        [ 0.6218,  0.7571],
        [-0.9056, -0.2783]])
```

The loss is pleasingly low, we have few false positives and negatives, and the ones we do have are close enough to the unit circle that we can forgive our model these minor transgressions.

### Conclusion

It's worth repeating that the example given above is very much contrived -- it solves a simple problem (and likely overfits!), on a tiny dataset, that runs in a few seconds on contemporary hardware. Nevertheless, I found putting it together helped solidify some key concepts, particularly around the dimensions involved in a network's weights and biases.


[^architecture]: I make no claims as to how "good" an architecture this is!
[^dimensions]: Note that has number of rows equal to the size of a single training data point (in this case, $2$), and number of columns equal to the number of neurons ($10$) chosen for this layer. In general, due to the rules of matrix multiplication, the number of rows in the weights of one layer must be equal to the number of outputs of the preceding layer (here we can consider the "zeroth" layer as the one which outputs our training examples). Note also in this post our training examples are encoded as rows, and so we multiply on the right with our weights. If our training examples were encoded as columns, we'd be multiplying on the left with our weights, and our weights would be transposed -- this is probably a good exercise!
[^broadcasting]: We lean on PyTorch's [broadcasting semantics][pytorch-broadcasting] here. Strictly, it doesn't makes sense to add a vector to a matrix, but with broadcasting, PyTorch can interpret our biases in a sensible way when performing addition.
[^pytorch-disclaimer]: I have deliberately implemented this example using tensors only, and not any of PyTorch's higher order convenience classes, purely in order to illustrate the mathematics at play.

[pytorch]: https://pytorch.org/
[pytorch-broadcasting]: https://pytorch.org/docs/stable/notes/broadcasting.html
[backprop]: https://en.wikipedia.org/wiki/Backpropagation