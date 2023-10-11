---
title: "A lightweight pub-sub framework"
date: 2023-10-05T22:58:18+01:00
draft: true
---

I've been playing around with the idea of building a lightweight framework for sending messages from a running process to a browser. Nothing Web Scale, as few moving parts as possible.

The rough idea I have is as follows:

* Producers emit messages down a Unix socket according to a simple protocol -- something like `| msg_length | num_fields | field_0_size | ... | field_n_size | field_0_data | ... | field_n_data |`.
* A daemon running on the same host has a receiving thread which listens for these messages and places them onto a ring buffer. Another sending thread listens for incoming websocket connections, reads from the ring buffer and sends the byte content as-is (maybe with the message length stripped). Some bookkeeping is required to make the threads play nicely with each other.
* On the browser, the received messages are parsed, and finally acted upon via callbacks.

I've built a proof of concept which mostly does all of this. As I write this, only one kind of message is supported, but I don't think it would be too much work to define messages as eg. Protobufs, and generate the sending and receiving code out of them.