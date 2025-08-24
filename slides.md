---
theme: seriph
background: https://images.pexels.com/photos/14169336/pexels-photo-14169336.jpeg?_gl=1*1puzlha*_ga*Nzk5Njk1MzQ5LjE3NTYwMjY2MTY.*_ga_8JE65Q40S6*czE3NTYwMjY2MTYkbzEkZzEkdDE3NTYwMjY2NjQkajEyJGwwJGgw
class: 'text-center p-4'
highlighter: shiki
lineNumbers: false
info: |
  ## Distributed Representations Beyond Words and Characters
  Sentence and Paragraph Embeddings in NLP Pipeline
drawings:
  persist: false
css: unocss
style: |
  .slidev-layout {
    padding: 1.5rem !important;
  }

  .slidev-page {
    padding: 1rem !important;
  }

  h1, h2, h3 {
    margin-bottom: 0.75rem !important;
  }

  .grid {
    gap: 0.75rem !important;
  }

  ul, ol {
    margin: 0.25rem 0 !important;
  }

  .text-xs {
    font-size: 0.7rem !important;
    line-height: 1.2 !important;
  }

  .text-sm {
    font-size: 0.8rem !important;
    line-height: 1.3 !important;
  }

  pre {
    margin: 0.5rem 0 !important;
    font-size: 0.75rem !important;
  }

  table {
    font-size: 0.75rem !important;
    margin: 0.5rem 0 !important;
  }

  .p-3 {
    padding: 0.5rem !important;
  }

  .mt-4, .mt-6 {
    margin-top: 0.75rem !important;
  }

  .space-y-4 > * + * {
    margin-top: 0.5rem !important;
  }

  .space-y-1 > * + * {
    margin-top: 0.2rem !important;
  }

  .space-y-3 > * + * {
    margin-top: 0.5rem !important;
  }

  .compact-grid {
    gap: 0.5rem !important;
  }

  .compact-text {
    font-size: 0.7rem !important;
    line-height: 1.1 !important;
  }

  .ultra-compact {
    font-size: 0.65rem !important;
    line-height: 1 !important;
    margin: 0.1rem 0 !important;
  }

  .compact-card {
    padding: 0.4rem !important;
    margin: 0.2rem 0 !important;
  }

  .mini-text {
    font-size: 0.6rem !important;
    line-height: 0.9rem !important;
    margin: 0.1rem 0 !important;
  }

  .tiny-grid {
    gap: 0.4rem !important;
  }

  .micro-card {
    padding: 0.3rem !important;
    margin: 0.15rem 0 !important;
  }

  /* Ensure content doesn't overflow */
  .slidev-layout .grid {
    height: auto !important;
    min-height: auto !important;
  }

  /* Better spacing for list items */
  li {
    margin: 0.15rem 0 !important;
  }

  /* Ensure code blocks are readable */
  .shiki {
    font-size: 0.7rem !important;
    line-height: 1.3 !important;
  }

  /* Better spacing for cards and boxes */
  .border {
    margin: 0.3rem 0 !important;
  }

  /* Ensure tables fit properly */
  table td, table th {
    padding: 0.3rem 0.5rem !important;
    font-size: 0.7rem !important;
  }

  /* Reduce heading sizes */
  h1 {
    font-size: 1.8rem !important;
  }

  h2 {
    font-size: 1.4rem !important;
  }

  h3 {
    font-size: 1.2rem !important;
  }

  h4 {
    font-size: 1rem !important;
  }
---

# Distributed Representations Beyond Words and Characters

## Sentence/Paragraph Embeddings in NLP Pipeline

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-4 py-2 rounded cursor-pointer bg-blue-600 text-white hover:bg-blue-700 transition-colors">
    Start Presentation
  </span>
</div>

<div class="abs-br m-6 text-sm opacity-60">
  Advanced NLP • Text Representation
</div>

---

# The NLP Pipeline: Text Representation Stage

<div class="grid grid-cols-2 gap-4 mt-3">

<div>

## Traditional Approach
- Tokenization → Words/Characters
- Word Embeddings (Word2Vec, GloVe)
- **Limitations:** Fixed vocabulary, no context

</div>

<div>

## Modern Approach
- Contextual Understanding
- Variable-length sequences
- Semantic composition
- Beyond individual tokens

</div>

</div>

<div class="mt-4 p-3 bg-blue-50 rounded-lg text-sm">
<strong>Key Challenge:</strong> How do we represent entire sentences and paragraphs as dense vectors while preserving semantic meaning?
</div>

---

# From Words to Sentences: The Challenge

<div class="mt-2">

## The Compositional Problem

```python {all|1-5|7-9}
# Word-level embeddings
word_vectors = {
    "cat": [0.2, 0.8, 0.1],
    "sat": [0.5, 0.3, 0.7],
}

# Simple averaging loses context
sentence = "The cat sat on the mat"
simple_avg = average(word_vectors)
```

<div class="mt-4 p-3 bg-red-50 border border-red-200 rounded text-sm">
<strong>Problem:</strong> Simple averaging loses word order, syntax, and semantic relationships
</div>

</div>

---

# Issues with Simple Approaches

<div class="grid grid-cols-3 tiny-grid mt-3">

<div class="micro-card bg-red-50 border border-red-200 rounded">
<h4 class="text-red-700 font-bold compact-text">Simple Averaging Issues</h4>
<ul class="mt-2 ultra-compact">
<li>Loses word order</li>
<li>Ignores syntax</li>
<li>No semantic composition</li>
</ul>
</div>

<div class="micro-card bg-yellow-50 border border-yellow-200 rounded">
<h4 class="text-yellow-700 font-bold compact-text">Bag of Words Problem</h4>
<ul class="mt-2 ultra-compact">
<li>"Dog bites man" ≈ "Man bites dog"</li>
<li>Same vectors, different meaning</li>
<li>Context-free representation</li>
</ul>
</div>

<div class="micro-card bg-green-50 border border-green-200 rounded">
<h4 class="text-green-700 font-bold compact-text">What We Need</h4>
<ul class="mt-2 ultra-compact">
<li>Preserve semantics</li>
<li>Consider context</li>
<li>Maintain relationships</li>
</ul>
</div>

</div>

<div class="mt-4 p-3 bg-blue-50 rounded-lg text-sm">
<strong>Solution:</strong> We need methods that can capture semantic meaning while preserving contextual relationships
</div>

---

# What are Sentence Embeddings?

<div class="grid grid-cols-2 gap-4 mt-2">

<div>

## Definition
Dense vector representations that capture semantic meaning of entire sentences in fixed-dimensional space.

## Key Properties
- Fixed dimensionality (256, 512, 768)
- Semantic similarity preserved
- Context-aware representations
- Task-agnostic or task-specific

</div>

<div>

## Mathematical Representation

```text
Sentence: "The weather is beautiful today"
         ↓
Encoder Function f(·)
         ↓
Vector: [0.23, -0.45, 0.67, ...]
        ∈ ℝᵈ (d = embedding dimension)
```

<div class="mt-3 p-3 bg-blue-50 rounded text-sm">
<strong>Semantic Similarity:</strong><br>
Similar sentences → Similar vectors<br>
cosine(v₁, v₂) ≈ 1 if sentences are similar
</div>

</div>

</div>

---

# Methods for Creating Sentence Embeddings - Part 1

<div class="grid grid-cols-2 gap-4 mt-2">

<div>

## 1. Aggregation Methods
- Simple Averaging
- Weighted Averaging (TF-IDF)
- Max/Min Pooling

```python
# Weighted average example
def weighted_embedding(words, weights):
    return sum(w * embed(word) 
              for word, w in zip(words, weights))
```

</div>

<div>

## 2. Sequential Models
- RNNs/LSTMs
- Last hidden state as embedding
- Bidirectional processing

```python
# LSTM sentence embedding
lstm = LSTM(hidden_size=256)
hidden_states = lstm(word_embeddings)
sentence_embedding = hidden_states[-1]
```

</div>

</div>

---

# Methods for Creating Sentence Embeddings - Part 2

<div class="grid grid-cols-2 gap-4 mt-2">

<div>

## 3. Transformer-based
- Self-attention mechanisms
- BERT, RoBERTa ([CLS] token)
- Sentence-BERT (fine-tuned)

```python
# BERT sentence embedding
sentence = "NLP is amazing"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)
embedding = outputs.last_hidden_state[:, 0, :]
```

</div>

<div>

## 4. Specialized Architectures
- Universal Sentence Encoder
- InferSent
- Quick Thoughts
- SimCSE

```python
# Universal Sentence Encoder
import tensorflow_hub as hub
encoder = hub.load("universal-sentence-encoder")
embeddings = encoder(["Hello world"])
```

</div>

</div>

---

# Popular Models: Universal Sentence Encoder

<div class="grid grid-cols-2 gap-4 mt-3">

<div>

## Architecture & Features
<div class="p-3 border border-blue-200 bg-blue-50 rounded-lg">
<ul class="text-sm space-y-1">
<li><strong>Developer:</strong> Google Research</li>
<li><strong>Architecture:</strong> Transformer + Deep Averaging Network</li>
<li><strong>Dimensions:</strong> 512</li>
<li><strong>Languages:</strong> Multilingual support</li>
<li><strong>Training:</strong> Multiple tasks (SNLI, STS, etc.)</li>
</ul>
</div>

</div>

<div>

## Strengths & Use Cases
<div class="p-3 border border-green-200 bg-green-50 rounded-lg">
<ul class="text-sm space-y-1">
<li><strong>Fast inference:</strong> Optimized for production</li>
<li><strong>General purpose:</strong> Works across domains</li>
<li><strong>Easy integration:</strong> TensorFlow Hub</li>
<li><strong>Good baseline:</strong> Solid performance</li>
<li><strong>Multilingual:</strong> 16+ languages</li>
</ul>
</div>

</div>

</div>

---

# Popular Models: Sentence-BERT (SBERT)

<div class="grid grid-cols-2 gap-4 mt-3">

<div>

## Architecture & Features
<div class="p-3 border border-green-200 bg-green-50 rounded-lg">
<ul class="text-sm space-y-1">
<li><strong>Developer:</strong> UKP Lab</li>
<li><strong>Architecture:</strong> BERT with Siamese network</li>
<li><strong>Dimensions:</strong> 768 (configurable)</li>
<li><strong>Training:</strong> Optimized for similarity tasks</li>
<li><strong>Fine-tuning:</strong> Task-specific adaptation</li>
</ul>
</div>

</div>

<div>

## Key Advantages
<div class="p-3 border border-purple-200 bg-purple-50 rounded-lg">
<ul class="text-sm space-y-1">
<li><strong>Speed:</strong> 65x faster than BERT pairs</li>
<li><strong>Quality:</strong> State-of-the-art similarity</li>
<li><strong>Flexibility:</strong> Multiple model sizes</li>
<li><strong>Community:</strong> Extensive model zoo</li>
<li><strong>Production-ready:</strong> Optimized inference</li>
</ul>
</div>

</div>

</div>

---

# Popular Models: InferSent

<div class="grid grid-cols-2 gap-4 mt-3">

<div>

## Architecture & Features
<div class="p-3 border border-purple-200 bg-purple-50 rounded-lg">
<ul class="text-sm space-y-1">
<li><strong>Developer:</strong> Facebook Research</li>
<li><strong>Architecture:</strong> BiLSTM with max pooling</li>
<li><strong>Dimensions:</strong> 4096</li>
<li><strong>Training:</strong> Stanford Natural Language Inference</li>
<li><strong>Focus:</strong> Transfer learning</li>
</ul>
</div>

</div>

<div>

## Performance Characteristics
<div class="p-3 border border-orange-200 bg-orange-50 rounded-lg">
<ul class="text-sm space-y-1">
<li><strong>Transfer learning:</strong> Good downstream performance</li>
<li><strong>Interpretability:</strong> Attention mechanisms</li>
<li><strong>Robustness:</strong> Handles various text types</li>
<li><strong>Research focus:</strong> Academic applications</li>
</ul>
</div>

</div>

</div>

---

# Model Performance Comparison

<div class="mt-3">

## Benchmark Results

| Model | Dimensions | Speed | STS Score | Best Use Case |
|-------|------------|-------|-----------|---------------|
| USE | 512 | Fast | 0.78 | General purpose |
| SBERT | 768 | Very Fast | 0.85 | Similarity search |
| InferSent | 4096 | Moderate | 0.75 | Transfer learning |
| SimCSE | 768 | Fast | 0.84 | Contrastive learning |

<div class="mt-4 grid grid-cols-2 gap-4">

<div class="p-3 bg-blue-50 rounded-lg text-sm">
<strong>Speed Comparison:</strong><br>
SBERT: 65x faster than BERT pairs<br>
USE: Optimized for production<br>
InferSent: Moderate inference time
</div>

<div class="p-3 bg-green-50 rounded-lg text-sm">
<strong>Quality Metrics:</strong><br>
STS = Semantic Textual Similarity<br>
Higher scores = better similarity<br>
Benchmark: STS-B dataset
</div>

</div>

</div>

---

# Paragraph Embeddings - Introduction

<div class="mt-2">

## What are Paragraph Embeddings?

Dense vector representations for entire documents or paragraphs that capture semantic meaning beyond individual sentences.

## Key Characteristics
- **Variable Length Handling:** Process documents of any size
- **Hierarchical Structure:** Capture both local and global context
- **Semantic Coherence:** Maintain meaning across long text spans
- **Document-Level Features:** Beyond sentence-level understanding

<div class="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm">
<strong>Challenge:</strong> How do we maintain semantic coherence across long documents while capturing both local details and global structure?
</div>

</div>

---

# Doc2Vec Approaches

<div class="grid grid-cols-2 gap-4 mt-3">

<div>

## PV-DM (Distributed Memory)
```python
# Predict word given context + paragraph vector
P(word | context_words, paragraph_id)
```

**Characteristics:**
- Uses paragraph vector as additional context
- Similar to Word2Vec CBOW
- Better for smaller datasets
- Preserves word order information

</div>

<div>

## PV-DBOW (Distributed Bag of Words)
```python
# Predict words given paragraph vector
P(words | paragraph_id)
```

**Characteristics:**
- Ignores word order in context
- Similar to Word2Vec Skip-gram
- Faster training, good performance
- More efficient for large corpora

</div>

</div>

<div class="mt-4 p-3 bg-blue-50 rounded-lg text-sm">
<strong>Best Practice:</strong> Combine both PV-DM and PV-DBOW for optimal results
</div>

---

# Hierarchical Attention Networks

<div class="mt-2">

## Two-Level Attention Mechanism

<div class="grid grid-cols-2 gap-4 mt-3">

<div>

### Word-Level Attention
- Identifies important words within sentences
- Creates sentence representations
- Preserves local context

```python
# Word attention example
word_attention = attention(words)
sentence_vectors = aggregate(word_attention)
```

</div>

<div>

### Sentence-Level Attention
- Identifies important sentences in documents
- Creates document representations
- Captures global structure

```python
# Sentence attention example
sentence_attention = attention(sentence_vectors)
document_vector = aggregate(sentence_attention)
```

</div>

</div>

## Complete Process
1. **Words → Sentences:** Word-level attention aggregates words into sentence vectors
2. **Sentences → Document:** Sentence-level attention aggregates sentences into document vector
3. **Hierarchical Understanding:** Captures both local and global semantic information

</div>

---

# Modern Transformer-Based Approaches

<div class="grid grid-cols-2 gap-4 mt-2">

<div>

## Long-Sequence Transformers

### Longformer
- **Sparse attention patterns**
- Linear complexity O(n)
- Sliding window + global attention
- Up to 4,096 tokens

### BigBird
- **Random + global + local attention**
- Theoretical guarantees
- Efficient for long documents
- Graph-based attention

</div>

<div>

## Hierarchical BERT Approaches

### Document Chunking
- Process documents in overlapping chunks
- Combine chunk representations
- Maintain global context across chunks

### Sentence-Level Processing
- Encode sentences individually
- Apply document-level transformer
- Preserve hierarchical structure

</div>

</div>

<div class="mt-4 p-3 bg-purple-50 border border-purple-200 rounded-lg text-sm">
<strong>Trade-off:</strong> Computational efficiency vs. representation quality vs. maximum sequence length
</div>

---

# Applications: Information Retrieval & Search

<div class="grid grid-cols-2 gap-4 mt-2">

<div>

## Semantic Search Systems
- **Query understanding:** Convert queries to embeddings
- **Document ranking:** Similarity-based retrieval
- **Cross-lingual search:** Multilingual embeddings
- **Personalization:** User-specific representations

## Question-Answering
- **Passage retrieval:** Find relevant text segments
- **Answer extraction:** Locate specific information
- **Context understanding:** Maintain conversation state

</div>

<div>

## Implementation Example

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

corpus = [
    "ML is a subset of AI",
    "Deep learning uses neural networks",
    "NLP handles text processing",
    "Computer vision processes images"
]
# Encode corpus
corpus_embeddings = model.encode(corpus)
# Search query
query = "What is artificial intelligence?"
query_embedding = model.encode([query])
# Find similarities
similarities = np.dot(query_embedding, corpus_embeddings.T)
top_idx = np.argmax(similarities)
print(f"Best match: {corpus[top_idx]}")
```

</div>

</div>

---

# Applications: Text Classification & Analysis

<div class="grid grid-cols-2 gap-4 mt-2">

<div>

## Classification Tasks

### Sentiment Analysis
- Movie reviews, product feedback
- Social media monitoring
- Customer satisfaction

### Topic Classification
- News categorization
- Email routing
- Content organization

### Intent Detection
- Chatbot understanding
- Voice assistants
- Customer service automation

</div>

<div>

## Clustering & Similarity

### Document Clustering
- Automatic topic discovery
- Content organization
- Research paper grouping

### Plagiarism Detection
- Academic integrity
- Content originality
- Copyright protection

### Recommendation Systems
- Content-based filtering
- Similar article suggestions
- Personalized feeds

</div>

</div>

---

# Technical Challenges

<div class="grid grid-cols-2 gap-4 mt-2">

<div class="space-y-3">

## Computational Complexity

<div class="p-3 bg-red-50 border border-red-200 rounded">
<h4 class="font-bold text-red-700 compact-text">Attention Complexity</h4>
<ul class="mt-2 ultra-compact">
<li>Quadratic complexity O(n²)</li>
<li>Memory requirements scale rapidly</li>
<li>GPU memory limitations</li>
</ul>
</div>

<div class="p-3 bg-orange-50 border border-orange-200 rounded">
<h4 class="font-bold text-orange-700 compact-text">Training Resources</h4>
<ul class="mt-2 ultra-compact">
<li>Large datasets required</li>
<li>Computational cost</li>
<li>Energy consumption</li>
</ul>
</div>

</div>

<div class="space-y-3">

## Representation Quality

<div class="p-3 bg-blue-50 border border-blue-200 rounded">
<h4 class="font-bold text-blue-700 compact-text">Information Loss</h4>
<ul class="mt-2 ultra-compact">
<li>Fixed-size bottleneck</li>
<li>Loss of fine-grained details</li>
<li>Compression artifacts</li>
</ul>
</div>

<div class="p-3 bg-purple-50 border border-purple-200 rounded">
<h4 class="font-bold text-purple-700 compact-text">Domain Adaptation</h4>
<ul class="mt-2 ultra-compact">
<li>General vs. domain-specific models</li>
<li>Transfer learning challenges</li>
<li>Out-of-domain performance</li>
</ul>
</div>

</div>

</div>

---

# Evaluation and Metrics

<div class="grid grid-cols-2 gap-4 mt-2">

<div>

## Intrinsic Evaluation

### Semantic Textual Similarity (STS)
- Pearson correlation with human judgments
- STS-B benchmark dataset
- Cross-lingual STS tasks

### Clustering Metrics
- Silhouette score
- Adjusted Rand Index
- Normalized Mutual Information

</div>

<div>

## Extrinsic Evaluation

### Downstream Task Performance
- Classification accuracy
- Information retrieval metrics
- Question-answering performance

### Practical Metrics
- Inference speed
- Memory usage
- Scalability measures

</div>

</div>

<div class="mt-4 p-3 bg-gray-50 rounded-lg text-center text-sm">
<strong>Key Trade-off:</strong> Expressiveness vs. Efficiency vs. Generalizability
</div>

---

# Future Directions and Research

<div class="grid grid-cols-2 gap-4 mt-2">

<div>

## Emerging Approaches

### Contrastive Learning
- SimCSE, ConSERT
- Self-supervised training
- Better representation quality

### Multimodal Embeddings
- Text + image representations
- Cross-modal understanding
- Unified embedding spaces

</div>

<div>

## Technical Improvements

### Efficiency Optimizations
- Sparse attention mechanisms
- Knowledge distillation
- Quantization techniques

### Better Architectures
- Retrieval-augmented generation
- Memory-efficient transformers
- Adaptive computation

</div>

</div>

<div class="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg text-sm">
<strong>Future Focus:</strong> More efficient, more capable, and more generalizable sentence and paragraph representations
</div>

---
layout: center
class: text-center
---

# Thank You!

Questions & Discussion

<div class="mt-8 space-y-4">
  <div class="text-lg opacity-80">
    Distributed Representations Beyond Words and Characters
  </div>
  <div class="text-sm opacity-60">
    Sentence/Paragraph Embeddings in NLP Pipeline
  </div>
</div>

<div class="mt-12 flex justify-center space-x-8 text-sm opacity-60">
  <span>🔬 Advanced NLP</span>
  <span>🧠 Semantic Understanding</span>
  <span>📊 Text Representation</span>
</div>