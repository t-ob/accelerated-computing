---
title: "Notes on neural networks: initialisation and normalisation techniques"
date: 2023-11-10T22:57:57+01:00
draft: false
math: true
---

### Untamed variances

By construction, the gradients of one layer in a neural network are dependent on the outputs of neurons in a previous layer. A consequence of this is that one layer's output might cause the preactivations of the next to fall within a hard-to-train regime. For example, consider the network with a single hidden layer with the following shape:

- Inputs have 200 dimensions
- Hidden layer has both fan-in and fan-out of 1000
- Outputs have 100 dimensions

A naive implementation (ignoring biases, to keep things simple), which draws weights from the standard random normal distribution might look like the following:

```python
import torch
import torch.nn.functional as F


torch.manual_seed(0)

fan_in = 200
hidden_dim = 1000
fan_out = 100

W_in = torch.randn((fan_in, hidden_dim), requires_grad=True)
W_hidden = torch.randn((hidden_dim, hidden_dim), requires_grad=True)
W_out = torch.randn((hidden_dim, fan_out), requires_grad=True)
```

Let's say further that we're trying to classify some inputs 32-at-a-time, so for non-linearities it seems reasonable to use ReLU for the input and hidden layers, and optimise cross entropy loss:

```python
batch_size = 32

X = torch.randn((batch_size, fan_in))

preactivations_in = X @ W_in
activations_in = F.relu(preactivations_in)

preactivations_hidden = activations_in @ W_hidden
activations_hidden = F.relu(preactivations_hidden)

logits = activations_hidden @ W_out

loss = F.cross_entropy(logits, Y)

loss.backward()
```

Note that we don't need to call `F.softmax` on our logits as `F.cross_entropy` will do that for us.

This certainly runs, but we have a problem -- our output layer is going to have a hard time learning:

```python
(W_out.grad == 0).float().sum()
```

```
tensor(78280.)
```

Over three quarters of its gradients are zero, so we can't expect it to learn very quickly.

### The problem

So what's going on? One useful lens to view the problem with is by considering variances. Our parameters are initialised by drawing from the standard normal distribution, so have zero mean and unit variance. But what happens as our inputs are transformed into logits as they flow through the network?

```python
print(f"{preactivations_in.var()=}")
print(f"{activations_in.var()=}")
print(f"{preactivations_hidden.var()=}")
print(f"{activations_hidden.var()=}")
print(f"{logits.var()=}")
```

```
preactivations_in.var()=tensor(194.2290, grad_fn=<VarBackward0>)
activations_in.var()=tensor(66.4669, grad_fn=<VarBackward0>)
preactivations_hidden.var()=tensor(98113.4062, grad_fn=<VarBackward0>)
activations_hidden.var()=tensor(33395.0352, grad_fn=<VarBackward0>)
logits.var()=tensor(46520924., grad_fn=<VarBackward0>)
```

The $(i,j)$-th entry of the matrix product $C = AB$ is given by the linear combination

$$
c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}.
$$

If we choose our $a$s and $b$s to be independent random variables, the variance of our $c$s becomes the sum of the variances of products:

$$
Var[c_{ij}] = \sum_{k=1}^{n} Var[a_{ik}]Var[b_{kj}] + Var[a_{ik}]E[b_{kj}]^{2} + Var[b_{kj}]E[a_{ik}]^{2}.
$$ 

In particular, as we go deeper in our network, our variance will typically grow, only shrinking when going through a ReLU.

By the time we output our logits, the variance has blown up, and when a softmax is performed during the calculation for cross entropy loss, we are left with essentially one-hot encoded vectors:

```python
logits.softmax(dim=-1)[0]
```

```
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], grad_fn=<SelectBackward0>)
```

The untrained model already has total confidence in its (most likely incorrect) predictions, and it's going to be difficult to convince it otherwise.

### The solution(s)

Luckily for us there are a few methods we can bring to bear to tackle this issue. The most straightforward way is simply to take some care in how we initialise our network parameters. We'll go into what that means shortly. More sophisticated techniques also exist, where special layers -- whose sole purpose is to regulate the growth of otherwise unchecked variance -- are inserted into the network. We'll take a look at one such kind of layer at the end of this article.

#### Parameter initialisation

In a [paper from 2015][arxiv-he-et-al][^nod-to-xavier], the authors show that in networks governed by ReLU activations, initialising each network weight with values drawn from a normal distribution with mean zero and variance of $\frac{2}{\text{fan\\_in}}$ will preserve the variances of each layer's outputs as values flow through the network[^or-fan-out].

To see why, let's derive a formula for the variance of any given layer's activations.

Let $y_l$, $W_l$, $x_l$, and $b_l$ be the outputs, weights, inputs, and bias (respectively) for the $l$-th layer in a network. Assume $W_l$ has fan-in of $n_{l,\text{in}}$ and fan-out of $n_{l,\text{out}}$, so that $W_l$ is a $n_{l,\text{out}} \times n_{l,\text{in}}$ matrix, the input $x_l$ is a $n_{l,\text{in}} \times 1$ vector, and we have

$$
y_l = W_l \cdot x_l + b_l = W_l \cdot \text{ReLU}(y_{l-1}) + b_l.
$$

Let's assume further that the values of $y_l$, $W_l$ and $x_l$ are drawn from mutually independent random variables, and that the $b_l$ are zero at initialisation.  Then, with apologies in advance for the abuse of notation that follows (the $Var$ in the expression in the middle refers to the variance of the matrix product, and the $Var$ in the expression on the right refers to the variance of the product of the random variables that $W_l$ and $x_l$ draw from), the variance of $y_l$ is given by

$$
Var\[y_l\] = Var\[ W_l \cdot x_l \] = n_{l,\text{in}} Var\[ W_l x_l \].
$$

The variance $Var\[ W_l x_l \]$ is given by

$$
Var\[ W_l x_l \] = Var\[ W_l \] Var\[ x_l \] + E\[ W_l \]^2 Var\[ x_l \] + E\[ x_l \]^2 Var\[ W_l \]
$$

and if we impose the condition that $W_l$ has zero mean, then the middle term cancels, so that

$$
Var\[ W_l x_l \] = Var\[ W_l \] ( Var\[ x_l \] + E\[ x_l \]^2 ) = Var\[ W_l \] E\[ x_l^2 \] 
$$

and

$$
Var\[y_l\] = n_{l,\text{in}} Var\[ W_l \] E\[ x_l^2 \].
$$

On the other hand, if we let $W_{l - 1}$ have zero mean and also be symmetric around zero, then the same is true for $y_{l-1}$ and we have

$$
Var\[ y_{l - 1} \] = 2 E\[ y_{l - 1}^2 | y_{l - i} > 0 \] P\[ y_{l - i} > 0 \]
$$

but as $x_l = \text{ReLU} (y_{l - 1}) = \text{max}(y_{l - 1}, 0)$, we have

$$
E\[x_l^2\] = E\[y_{l - 1}^2 | y_{l - 1} > 0 \]P\[ y_{l - 1} > 0 ] = \frac{1}{2} Var\[ y_{l - 1} \]
$$

so that

$$
Var\[y_l\] = \frac{1}{2} n_{l,\text{in}} Var\[ W_l \] Var\[ y_{l - 1} \].
$$

Thus, with $L$ layers, expanding the recurrence relation gives a formula for the final activations:

$$
Var \[ y_L \] = Var \[ y_1 ] ( \prod_{l=2}^L \frac{1}{2} n_{l,\text{in}} Var\[ y_{l - 1} \] )
$$

In particular, if we want our variances to remain stable throughout, it suffices to make each term in the above product equal to one. Or, in other words, set

$$
Var\[ y_{l - 1} \] = \frac{2}{n_{l,\text{in}}}.
$$

We can see this in action by making some small changes to our above code. Let's make the following change:

```python
# Old
# W_in = torch.randn((fan_in, hidden_dim), requires_grad=True)
# W_hidden = torch.randn((hidden_dim, hidden_dim), requires_grad=True)
# W_out = torch.randn((hidden_dim, fan_out), requires_grad=True)

# New
W_in = torch.empty(fan_in, hidden_dim, requires_grad=True)
W_hidden = torch.empty(hidden_dim, hidden_dim, requires_grad=True)
W_out = torch.empty(hidden_dim, fan_out, requires_grad=True)

torch.nn.init.kaiming_normal_(W_in, mode='fan_in', nonlinearity='relu')
torch.nn.init.kaiming_normal_(W_hidden, mode='fan_in', nonlinearity='relu')
torch.nn.init.kaiming_normal_(W_out, mode='fan_in', nonlinearity='relu')
```

Note that the `kaiming` in the above name is a reference to the primary author of the paper linked above. Running through and printing variances at the end gives:

```
preactivations_in.var()=tensor(0.3928, grad_fn=<VarBackward0>)
activations_in.var()=tensor(0.1341, grad_fn=<VarBackward0>)
preactivations_hidden.var()=tensor(0.4011, grad_fn=<VarBackward0>)
activations_hidden.var()=tensor(0.1358, grad_fn=<VarBackward0>)
logits.var()=tensor(4.2004, grad_fn=<VarBackward0>)
```

The activations have roughly the same variance, as desired, and the network has a more nuanced take on the logits it produces -- taking the first again as an example:

```
tensor([ 1.8816,  0.7903,  0.2349, -1.3213, -0.9871, -1.9959, -1.4037,  4.1994,
        -0.2390,  0.2372,  2.5905, -0.7748,  0.0910, -1.0113, -5.1495,  1.3632,
        -0.0707, -1.2270, -2.5708,  1.7268,  1.2458, -1.3190,  1.6976, -0.1668,
        -2.5448,  1.0397,  2.5572,  2.0769,  0.8277,  0.1977,  2.6933, -2.7683,
         0.3132, -2.7061,  1.5003, -1.9401,  0.4859,  1.1965, -3.9031,  1.3884,
        -5.1225,  0.6135, -0.5104,  1.1758, -0.0542,  2.1106, -1.7629, -0.7441,
        -2.9037, -1.7349,  2.6031, -2.2783, -0.2811, -1.7943, -0.8837,  3.9457,
         1.2524,  5.4557, -1.9571,  3.6366,  3.2720, -1.2876, -2.5694,  2.4767,
         2.8721, -2.0328, -4.0159, -0.6583,  3.1763,  0.7272, -1.0748, -2.4906,
         1.2876, -0.2730,  3.5285, -0.5165,  1.2238,  2.9884, -0.5808,  2.3573,
        -0.9532,  0.8772, -1.6501, -0.6264, -4.6388,  1.7694,  1.0911,  0.9606,
         1.7970, -2.8896,  1.2940, -1.1037,  1.0425, -1.6953,  2.8094, -1.3222,
         3.1458,  1.2113, -4.7665, -0.1857], grad_fn=<SelectBackward0>)
```

It's worth noting that, with the invention of other normalisation techniques (discussed below), this initialisation technique may not always be necessary, and you might find that a simple `torch.nn.Linear` module, which samples uniformly from the interval $( \frac{-1}{\sqrt{\text{fan\\_in}}} , \frac{1}{\sqrt{\text{fan\\_in}}} )$ is good enough for your needs.

#### Batch normalisation

We've seen that a considered approach to network initialisation can prevent variances from blowing up as the network gets deeper, but there are no guarantees that the distribution of each layer's activations won't change (possibly dramatically) over the course of training. This phenomena is dubbed "internal covariate shift" in a [paper from 2015][arxiv-ioffe-szegedy] that introduces the idea of batch normalisation.

The key idea in the paper is a simple network layer which does two things:

1. Normalises its inputs to have zero mean and unit variance across each dimension
2. Applies a linear function (with learnable parameters) to the normalised inputs

The paper goes on to show how employing batch normalisation they were able to match the previous best performance on ImageNet classification using a fraction of the training steps, and then go on to surpass it.

Let's break these steps down a bit. We are going to construct a network layer which takes inputs of dimension $n$, and outputs some normalised values of the same shape.

Let $X = (x_{i, j})$ be a batch of $m$ inputs (so $X$ is an $m \times n$ matrix). For each $j$, we can compute the mean $E\[x_j\]$ and variance $Var\[x_j\]$ across all $m$ examples in the batch. Let $\epsilon$ be some small constant (to protect against division by zero), and define $\hat{X}$ to be the matrix with the same shape as $X$, whose $(i,j)$-th entry $\hat{x}_{i, j}$ is given by

$$
\hat{x}\_{i, j} = \frac{x_{i, j} - E\[x_j\]}{\sqrt{Var\[x_j\] + \epsilon}}.
$$

Now $\hat{X}$ has zero mean and unit variance across each of its dimensions, but this normalisation would be too aggressive if we just called it a day and returned it as-is (the example given in the paper is roughly -- imagine these inputs were to be fed into a sigmoid activation; after this normalisation most values will now fall into the linear regime of the nonlinearity, making it hard to learn anything interesting).

The key idea in the paper is to augment the normalisation above with a linear scale and shift with learnable parameters $\gamma = (\gamma_1, \ldots , \gamma_n)$ and $\beta = (\beta_1, \dots \beta_n)$ to produce its output $\hat{Y}$, whose $(i,j)$-th entry $\hat{y}_{i,j}$ is given by

$$
\hat{y}\_{i,j} = \gamma\_j x\_{i,j} + \beta\_j
$$

By ensuring that "the transformation inserted in the network can represent the identity transform," we ensure we don't change what the network can represent.

One issue that batch normalisation introduces is that at inference time, the layer (as currently defined) cannot handle a single input example (ie. a batch of size one). One approach to resolve this is to compute the mean and variances over the entire dataset once after training and then use those at inference, or (more commonly), keep exponentially weighted moving averages of the running means and variances as training proceeds, then use those at inference time.

Here's an example of what the process looks like after a single pass:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# To guard against division by zero
eps = torch.tensor(1e-5)

# Learnable parameters
gamma = torch.ones((1, 100), requires_grad=True)
beta = torch.zeros((1, 100), requires_grad=True)

# EWMA parameter (not learned)
momentum = 0.001

# Running stats
X_mean_running = torch.zeros((1, 100))
X_std_running = torch.ones((1, 100))

# Batch with a large variance
X = torch.randn((32, 100)) * 100

# Batch 
X_mean = X.mean(dim=0, keepdim=True)
X_std = X.std(dim=0, keepdim=True)

# Normalised output
X_hat = gamma * ((X - X_mean) / (X_std + eps)) + beta

# Update running statistics
with torch.no_grad():
    X_mean_running = (1.0 - momentum) * X_mean_running + momentum * X_mean
    X_std_running = (1.0 - momentum) * X_std_running + momentum * X_std
```

And inspecting the statistics of `X_hat` gives us what we expect (for an untrained network):

```python
print(f"{X_hat.mean()=}")
print(f"{X_hat.var()=}")
```

```
X_hat.mean()=tensor(1.1921e-09, grad_fn=<MeanBackward0>)
X_hat.var()=tensor(0.9691, grad_fn=<VarBackward0>)
```

In practice, you should reach for the built-in `torch.nn.BatchNorm1d` (and its higher dimensional variants) when appropriate, but internally they will proceed more or less like the above. It's important also to set your model to evaluation mode (`.eval()` on a `torch.nn.Module`) at inference time, and training mode (the default, `.train()`) at training time, so the layer uses the appropriate statistics.

### Further reading

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (arXiv)][arxiv-he-et-al]
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (arXiv)][arxiv-ioffe-szegedy]
- [Neural Networks: Zero to Hero - Building makemore Part 3: Activations & Gradients, BatchNorm (YouTube)][youtube-nnzth]


[^or-fan-out]: The alternative variance $\frac{2}{\text{fan\\_out}}$ is presented as an equally good choice. The former preserves variances on the forward pass, and latter on the backward pass.

[^nod-to-xavier]: This paper builds on an earlier [paper from 2010][xavier-glorot] which performed a similar analysis on networks with sigmoid-like activations.


[arxiv-ioffe-szegedy]: https://arxiv.org/abs/1502.03167
[arxiv-he-et-al]: https://arxiv.org/abs/1502.01852v1
[xavier-glorot]: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
[youtube-nnzth]: https://www.youtube.com/watch?v=P6sfmUTpUmc