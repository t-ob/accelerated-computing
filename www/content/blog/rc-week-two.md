---
title: "RC: Week two"
date: 2023-10-01T12:17:57+01:00
draft: false
---

I'm writing this the weekend after my first fortnight at the Recurse Center. It feels like I'm beginning to find a rhythm, though I've noticed I can quite easily fall into a head-down, tunnel vision mode of programming, and think I should be engaging in some more pair programming. Something to aim for in the coming weeks.

My time so far has been spent either trying to learn new things, working on concrete projects, and getting to know my fellow batch-mates.

In terms of learning, I've been watching Andrej Karpathy's [excellent series][karpathy] on neural networks, and making my progress available on [GitHub][gh-nlp-course]. I found a small detour into exploring the [Weights & Biases][wandb] library when performing some light hyperparameter optimisation a highly worthwhile exercise. I've also been supplementing Karpathy's series of lectures with [deeplearning.ai's Deep Learning specialisation][dlai] -- from past experience I've found that when trying to learn new material, having two sources and perspectives to learn from can really help cement my understanding. There is a small but active group of us going through the material together.

I've also been working through Part 2 of [Crafting Interpreters][nystrom]. There was an existing group of attendees from the preceding batch (RC overlaps batches so that as one twelve-week batch hits its half way point, another batch begins) who were just starting, having just finished Part 1, so I'm tagging along. I've opted for C++ over the textbook C implementation for a couple of reasons:
- I can lean on its conveniences over C (eg. containers) when I feel like it.
- I'm trying to become a better C++ programmer anyway.
- It's close enough to C so that I don't have to perform too much mental gymnastics to translate (I started with OCaml but abandoned that attempt swiftly after realising that knowing almost nothing about neither compilers nor OCaml would make for poor progress).

We've just finished Chapter 17, and it feels pretty cool to be able to boot into a repl and be able to perform some basic arithmetic.

As for concrete projects, one idea I'm currently running with is building a proof of concept for a light-weight, self-hosted, reasonably (I hope) high performance system for shuttling bytes from one process to a browser in real time. I started following this thread while watching [the fourth Karpathy lecture][karpathy-4] where there is a lot of evaluate-cell-plot-in-matplotlib-update-cell-repeat going on in a Jupyter notebook. I am a big fan of rapid feedback in all forms, but sometimes you need feedback from a process that isn't a notebook. Of course, stdout is a good candidate, but limited in how much useful information can be crammed in there. The idea right now is to send data down a Unix socket to an agent which updates a ring buffer on one end on one thread, and pushes updates to clients connected over a websocket on another. The frontend doesn't exist yet, but that's for next week.

All in all, I am delighted with how the experience has been so far. I had some concerns going in about missing out on the IRL experience, but I am pleased to report that none of those have played-out -- RC is well equipped to support remote attendees, and the groups I've participated in have been nothing but welcoming to remote participants.

[karpathy]: https://karpathy.ai/zero-to-hero.html
[gh-nlp-course]: https://github.com/t-ob/nlp-course
[wandb]: https://wandb.ai
[dlai]: https://www.deeplearning.ai/courses/deep-learning-specialization/
[nystrom]: https://craftinginterpreters.com/
[karpathy-4]: https://www.youtube.com/watch?v=P6sfmUTpUmc