# Multi Trait SGNS

### Introduction

This repo includes code for a neural network for learning embeddings of items which include additional "side information". Paritcularly useful for tackling cold start problems in situations with increasing an increasing item space (e.g. retail, video recommendation, etc.)

### Inspiration

The code was adapted from the blueprint outlined in [1]. The paper itself is not especially specific about the neural network structure. Specific choices made:
1. Separate embeddings for target and context items
2. Separate vector allowed for defining weighted average embeddings between target and context
3. A single item's overall embedding can be extracted as the element-wise average of its target and context embedding

### Design Choices

Choice (1) was made to remain consistent with most implementations of Word2Vec whereby the increased complexity allowed by separating treatment of target/context allow for more powerful embeddings. Chocie (2) is in the same spirit as choice (1). Choice (3) seems a reasonable (and not novel) method for extracting a single embedding, although the target and context embeddings can be individually extracted if one wishes.

### To-Do
* Automate cpu/gpu flexibility
* Basic unit tests for individual operations
* End-to-end test for "forward"
* Functional test to ensure gradients are correctly passing back to embeddings

[[1] Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1803.02349)
