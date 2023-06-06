---
title: "Introduction"
slug: introduction
date: 2023-06-07T15:40:25+01:00
weight: 11
draft: false
---

## Introduction
Accelerated computing is a broad term. It describes the specialised hardware, software, and programming paradigms which can be employed to perform certain classes of computation more efficiently than is possible on a general-purpose CPU alone. 

This specialised hardware typically includes Graphics Processing Units (GPUs), Field Programmable Gate Arrays (FPGAs), Application-Specific Integrated Circuits (ASICs), and other forms of hardware accelerators. These accelerators are designed to handle specific types of computations more efficiently, particularly for workloads that involve parallel processing, like machine learning, high performance computing (HPC), and graphics rendering.

GPUs, for example, are excellent at performing floating-point operations quickly and in parallel, which makes them highly effective for tasks such as rendering graphics and performing machine learning algorithms. FPGAs can be reprogrammed to perform specific operations with high efficiency, while ASICs are custom-designed for a specific application or task.

The software for accelerated computing includes specialized libraries and APIs that allow programmers to take advantage of the features of the hardware accelerators. This includes software frameworks like CUDA for NVIDIA GPUs, OpenCL for various types of accelerators, and others.

This text currently limits its focus to NVidia GPUs and associated software, though this may change in the future.

We start this chapter with a brief introduction to the CUDA programming framework. We then use these key concepts to build a simple (but non-trivial) CUDA C++ program which performs its computation on a GPU. Finally, we will benchmark this program against comparable CPU-bound applications, as well as against an optimised reference implementation.