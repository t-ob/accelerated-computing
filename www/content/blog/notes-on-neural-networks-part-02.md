---
title: "Notes on neural networks: loss, gradient descent, and backpropagation"
date: 2023-10-10T22:57:57+01:00
draft: false
math: true
---

### Minimising loss via gradient descent

When we talk about neural networks, the loss of a given network architecture on a set of example inputs is a scalar value which represents how well the model does (or doesn't) fit those inputs[^loss-caveat]. Examples of loss functions include [root mean squared error][loss-rms] and [cross-entropy][loss-ce].

Training a neural network involves minimising some chosen loss function $ℒ$. It is an iterative process. The most straightforward way to go about this minimisation is via the technique known as gradient descent -- at a high level, given some initial parameters ${\theta = (\theta\_1, \ldots, \theta\_k)}$ of our network[^params], and a loss function $ℒ$, we can implement gradient descent by repeatedly performing the following steps:
1. Perform a forward pass of our network on a training data set $X$[^X-choice].
2. Measure the loss $ℒ(\hat{Y}, Y)$ of the output $\hat{Y}$ against training labels $Y$.
3. Compute the partial derivatives of $ℒ(\hat{Y}, Y)$ with respect to every network parameter  $\theta\_i$.
4. Adjust each $\theta\_i$ by subtracting from it a small multiple (the learning rate $\alpha$) of its partial derivative: ${\theta\_i  \gets \theta\_i - \alpha \cdot \frac{\delta ℒ(\hat{Y}, Y)}{\delta \theta\_i}}$.

If we've chosen a good network architecture, with good parameters, and haven't over-fitted our training dataset, the output of this process is a set of weights which define a function that can be applied to any input (from the training dataset or otherwise), and observe a sensible output on the other side. 

Let's unpack that a bit.

A neural network is defined by its parameters $\theta$ and its non-linearities (ReLU and its variants, tanh, etc.). It takes some input $X$ (either one or many examples), performs a forward pass, and spits out some output $Y$. We can also think of it in a different way -- given some fixed input and output $X$ and $Y$ (eg. a training dataset), and some loss function $ℒ$, a neural network can be seen as a mapping from its parameter space to the real numbers. The mapping takes the network parameters $\theta$ and applies the composition of the forward pass with the loss $ℒ$ to them. If our choices of loss function and non-linearities are differentiable, this composition is also differentiable, and we can perform gradient descent as described above.

Why does subtracting a small multiple of each parameter's partial derivative result in a reduced loss? Let's restrict ourselves to the single-variable case (which generalises to multi-variable but I think is conceptually easier to understand), and assume we have some differentiable real-valued function $f$ of a single variable. Then, its derivative $f^\prime(x)$ of $f$ at $x$ is defined as the limit

$$
f^\prime(x) = \lim_{t \to 0} \frac{f(x + t) - f(x)}{t}
$$

This means that, in a small neighbourhood of $x$, we have approximately

$$
t \cdot f^\prime(x) \approx f(x + t) - f(x)
$$

Rearranging, and taking $t$ to be $\alpha \cdot f^\prime(x)$ (for a sufficiently small choice of $\alpha > 0$), we have

$$
f(x + \alpha \cdot f^\prime(x)) \approx f(x) + \alpha \cdot (f^\prime(x))^2
$$

In other words, adding a sufficiently small multiple of $f^\prime(x)$ to $x$ will result in an increased (or equal, if the derivative is zero) value of $f$ (because $\alpha \cdot (f^\prime(x))^2$ is always non-negative). It follows then, that

$$
f(x - \alpha \cdot f^\prime(x)) \approx f(x) - \alpha \cdot (f^\prime(x))^2 \leq f(x)
$$

That is, at any given $x$, subtracting a small multiple of the derivative at $x$ moves to reduce the value of $f$. In the case of neural networks, the loss function $ℒ$ plays the role of $f$, and each parameter is analagous to an $x$.

### Backpropagation

How are these gradients calculated in practice? We could do it by hand -- the chain rule is well-understood, after all -- but this approach does not scale and would likely be error prone, to say nothing about being incredibly tedious. Instead, the process of backpropagation is employed. The basic idea follows this formulation of the chain rule: if we have $n$ variables $z\_{1} \ldots z\_{n}$ such that $z\_1 = z\_1(x)$ depends on $x$, and $z\_{i} = z\_{i}(z\_{i - 1})$ depends on $z\_{i - 1}$, then every $z\_{i}$ depends on $x$, and

$$
\frac{dz\_n}{dx} = \frac{dz\_{n}}{dz\_{n - 1}} \cdot \frac{dz\_{n - 1}}{dx} = \frac{dz\_{n}}{dz\_{n - 1}} \cdot \frac{dz\_{n - 1}}{dz\_{n - 2}} \cdot \enspace \cdots \enspace \cdot \frac{dz\_{1}}{dx}
$$

In the context of neural networks, we're interested in the case where the role of $x$ is played by some parameter of our network, $z\_n$ the loss function, and all the intermediate $z\_i$ representing the forward pass.

In the multivariate case, if we have $z\_n = z\_n(z\_{n-1, 1} , \cdots , z\_{n-1, k})$, where each $z\_{n-1, i}$ depends on some other variable $x$, then

$$
\frac{\delta z\_n}{\delta x} = \sum\_{i=1}^{k} \frac{\delta z\_{n}}{\delta z\_{n - 1, i}} \cdot \frac{\delta z\_{n - 1, i}}{\delta x} 
$$

Backpropagation is, at its core, the translation of this formula into code.

In practice, libraries like PyTorch and Tensorflow will maintain a computation graph of tensors, where the result of an operation on some tensors results in a tensor containing two things:

1. References to the input tensors it depends on.
2. A function which encodes its contribution to the backward pass, by updating the derivatives of its inputs according to the chain rule above.

When backpropagation is triggered on some final tensor `L` (with eg. `.backward()` in PyTorch), a backward pass begins -- the graph is traversed in reverse topological order (so that the derivatives typically referenced in the backward pass contributing functions in the graph are computed by the time they're needed) from `L` to its leaf nodes. At each node (ie. at each tensor), its contribution to the backward pass is computed, updating the derivatives of all of its immediate input tensors.


[^loss-caveat]: That is not to say that a lower loss always means a better model -- a model may have achieve a near-zero loss by wildly over-fitting its training data, performing poorly otherwise. Similarly, methods such as regularisation may increase loss while protecting against over-fitting, which may be a worthwhile tradeoff to make.
[^params]: Network parameters just means the set of all weights and all biases.
[^X-choice]: When $X$ is chosen to be the entire available training data, this is known as batch gradient descent. When $X$ is a single example, this is stochastic gradient descent. When $X$ is somewhere in the middle, ie. a small subset of training data, this is mini-batch gradient descent.

[loss-rms]: https://en.wikipedia.org/wiki/Root_mean_square
[loss-ce]: https://en.wikipedia.org/wiki/Cross-entropy