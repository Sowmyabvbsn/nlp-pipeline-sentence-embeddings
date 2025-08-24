---
theme: seriph
background: https://images.pexels.com/photos/6550258/pexels-photo-6550258.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080
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
    padding: 2rem !important;
  }

  .slidev-page {
    padding: 1.5rem !important;
  }

  h1, h2, h3 {
    margin-bottom: 1rem !important;
  }

  .grid {
    gap: 1rem !important;
  }

  ul, ol {
    margin: 0.5rem 0 !important;
  }

  .text-xs {
    font-size: 0.75rem !important;
    line-height: 1.3 !important;
  }

  .text-sm {
    font-size: 0.85rem !important;
    line-height: 1.4 !important;
  }

  pre {
    margin: 0.75rem 0 !important;
    font-size: 0.8rem !important;
  }

  table {
    font-size: 0.8rem !important;
    margin: 0.75rem 0 !important;
  }

  .p-3 {
    padding: 0.75rem !important;
  }

  .mt-4, .mt-6 {
    margin-top: 1rem !important;
  }

  .space-y-4 > * + * {
    margin-top: 0.75rem !important;
  }

  .space-y-1 > * + * {
    margin-top: 0.25rem !important;
  }

  .space-y-3 > * + * {
    margin-top: 0.75rem !important;
  }

  .compact-grid {
    gap: 0.75rem !important;
  }

  .compact-text {
    font-size: 0.75rem !important;
    line-height: 1.2 !important;
  }

  .ultra-compact {
    font-size: 0.7rem !important;
    line-height: 1.1 !important;
    margin: 0.2rem 0 !important;
  }

  .compact-card {
    padding: 0.5rem !important;
    margin: 0.3rem 0 !important;
  }

  .mini-text {
    font-size: 0.65rem !important;
    line-height: 1rem !important;
    margin: 0.1rem 0 !important;
  }

  .tiny-grid {
    gap: 0.5rem !important;
  }

  .micro-card {
    padding: 0.4rem !important;
    margin: 0.2rem 0 !important;
  }

  /* Ensure content doesn't overflow */
  .slidev-layout .grid {
    height: auto !important;
    min-height: auto !important;
  }

  /* Better spacing for list items */
  li {
    margin: 0.2rem 0 !important;
  }

  /* Ensure code blocks are readable */
  .shiki {
    font-size: 0.8rem !important;
    line-height: 1.4 !important;
  }

  /* Better spacing for cards and boxes */
  .border {
    margin: 0.5rem 0 !important;
  }

  /* Ensure tables fit properly */
  table td, table th {
    padding: 0.4rem 0.6rem !important;
    font-size: 0.75rem !important;
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

<div class="grid grid-cols-2 gap-4 mt-4">

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

# From Words to Sentences: The Compositional Problem

<div class="mt-2">

## The Challenge of Compositionality

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

<div class="grid grid-cols-3 gap-3 mt-4 text-sm">

<div class="p-3 bg-red-50 border border-red-200 rounded">
<h4 class="text-red-700 font-bold text-sm">Simple Averaging Issues</h4>
<ul class="mt-2">
<li>Loses word order</li>
<li>Ignores syntax</li>
</ul>
</div>

<div class="p-3 bg-yellow-50 border border-yellow-200 rounded">
<h4 class="text-yellow-700 font-bold text-sm">Bag of Words</h4>
<ul class="mt-2">
<li>"Dog bites man" ≈ "Man bites dog"</li>
<li>Same vectors, different meaning</li>
</ul>
</div>

<div class="p-3 bg-green-50 border border-green-200 rounded">
<h4 class="text-green-700 font-bold text-sm">Solution</h4>
<ul class="mt-2">
<li>Preserve semantics</li>
<li>Consider context</li>
</ul>
</div>

</div>

</div>

---

# What are Sentence Embeddings?

<div class="grid grid-cols-2 gap-4 mt-3">

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

## Math Representation

```text
Sentence: "The weather is beautiful today"
         ↓
Encoder Function f(·)
         ↓
Vector: [0.23, -0.45, 0.67, ...]
        ∈ ℝᵈ (d = embedding dimension)
```

<div class="mt-4 p-3 bg-blue-50 rounded text-sm">
<strong>Semantic Similarity:</strong>
Similar sentences → Similar vectors
cosine(v₁, v₂) ≈ 1 if sentences are similar
</div>

</div>

</div>

---

# Methods for Creating Sentence Embeddings

<div class="grid grid-cols-2 gap-4 mt-3">

<div>

## 1. Aggregation Methods
- Simple Averaging
- Weighted Averaging (TF-IDF)
- Max/Min Pooling

```python {all|1-3}
# Weighted average example
def weighted_embedding(words, weights):
    return sum(w * embed(word) for word, w in zip(words, weights))
```

## 2. Sequential Models
- RNNs/LSTMs
- Last hidden state as embedding
- Bidirectional processing

</div>

<div>

## 3. Transformer-based
- Self-attention mechanisms
- BERT, RoBERTa ([CLS] token)
- Sentence-BERT (fine-tuned)

```python {all|1-2|3-5}
# BERT sentence embedding
sentence = "NLP is amazing"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)
embedding = outputs.last_hidden_state[:, 0, :]
```

## 4. Specialized Architectures
- Universal Sentence Encoder
- InferSent
- Quick Thoughts

</div>

</div>

---

# Popular Sentence Embedding Models - Part 1

<div class="grid grid-cols-2 gap-4 mt-4">

<div>

## Universal Sentence Encoder
<div class="p-4 border border-blue-200 bg-blue-50 rounded-lg">
<ul class="text-sm space-y-2">
<li><strong>Developer:</strong> Google Research</li>
<li><strong>Architecture:</strong> Transformer + Deep Averaging Network</li>
<li><strong>Dimensions:</strong> 512</li>
<li><strong>Languages:</strong> Multilingual support</li>
<li><strong>Strengths:</strong> Fast inference, good general performance</li>
</ul>
</div>

</div>

<div>

## Sentence-BERT (SBERT)
<div class="p-4 border border-green-200 bg-green-50 rounded-lg">
<ul class="text-sm space-y-2">
<li><strong>Developer:</strong> UKP Lab</li>
<li><strong>Architecture:</strong> BERT with Siamese network</li>
<li><strong>Dimensions:</strong> 768</li>
<li><strong>Training:</strong> Optimized for similarity tasks</li>
<li><strong>Strengths:</strong> Very fast inference, excellent similarity</li>
</ul>
</div>

</div>

</div>

---

# Popular Sentence Embedding Models - Part 2

<div class="grid grid-cols-2 gap-4 mt-4">

<div>

## InferSent
<div class="p-4 border border-purple-200 bg-purple-50 rounded-lg">
<ul class="text-sm space-y-2">
<li><strong>Developer:</strong> Facebook Research</li>
<li><strong>Architecture:</strong> BiLSTM with max pooling</li>
<li><strong>Dimensions:</strong> 4096</li>
<li><strong>Training:</strong> Stanford Natural Language Inference</li>
<li><strong>Strengths:</strong> Good transfer learning performance</li>
</ul>
</div>

</div>

<div>

## Performance Comparison

| Model | Dimensions | Speed | STS Score | Best Use Case |
|-------|------------|-------|-----------|---------------|
| USE | 512 | Fast | 0.78 | General purpose |
| SBERT | 768 | Very Fast | 0.85 | Similarity search |
| InferSent | 4096 | Moderate | 0.75 | Transfer learning |

<div class="mt-4 p-3 bg-gray-50 rounded-lg text-sm">
<strong>Note:</strong> STS = Semantic Textual Similarity benchmark score
</div>

</div>

</div>

---

# Paragraph Embeddings - Introduction

<div>

## What are Paragraph Embeddings?

Dense vector representations for entire documents or paragraphs that capture semantic meaning beyond individual sentences.

## Key Characteristics
- **Variable Length Handling:** Can process documents of any size
- **Hierarchical Structure:** Captures both local and global context
- **Semantic Coherence:** Maintains meaning across long text spans
- **Document-Level Features:** Goes beyond sentence-level understanding

</div>

<div class="mt-6">

## Doc2Vec Approaches

<div class="grid grid-cols-2 gap-4 mt-4">

<div>

### PV-DM (Distributed Memory)
```python
# Predict word given context + paragraph vector
P(word | context_words, paragraph_id)
```
- Uses paragraph vector as additional context
- Similar to Word2Vec CBOW
- Better for smaller datasets

</div>

<div>

### PV-DBOW (Distributed Bag of Words)
```python
# Predict words given paragraph vector
P(words | paragraph_id)
```
- Ignores word order in context
- Similar to Word2Vec Skip-gram
- Faster training, good performance

</div>

</div>

</div>

---

# Modern Paragraph Embedding Approaches

<div class="grid grid-cols-2 gap-4 mt-3">

<div>

## Hierarchical Attention Networks
- **Word-level attention:** Important words in sentences
- **Sentence-level attention:** Important sentences in documents
- **Two-stage process:** Word → Sentence → Document

```python
# Hierarchical attention example
word_attention = attention(words)
sentence_vectors = aggregate(word_attention)
sentence_attention = attention(sentence_vectors)
document_vector = aggregate(sentence_attention)
```

</div>

<div>

## Transformer-Based Methods

### Long-Sequence Transformers
- **Longformer:** Sparse attention patterns
- **BigBird:** Random + global + local attention
- **LED:** Long document summarization

### Hierarchical BERT
- Process documents in chunks
- Combine chunk representations
- Maintain global context

</div>

</div>

<div class="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
<strong>Key Challenge:</strong> Maintaining semantic coherence across long documents while capturing both local details and global structure.
</div>

---

# Applications and Use Cases

<div class="grid grid-cols-2 gap-4 mt-3">

<div>

## Information Retrieval
- Semantic search engines
- Document ranking
- Question-answering systems
- Recommendation systems

## Text Classification
- Sentiment analysis
- Topic classification
- Intent detection
- Spam filtering

## Similarity & Clustering
- Document clustering
- Plagiarism detection
- Duplicate detection
- Content recommendation

</div>

<div>

## Semantic Search Example

```python {all|1-3|5-10|12-14|16-22}
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

corpus = [
    "ML is a subset of AI",
    "Deep learning uses networks",
    "NLP handles text",
    "CV processes images"
]

corpus_embeddings = model.encode(corpus)

query = "What is artificial intelligence?"
query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, corpus_embeddings)
top_idx = np.argmax(similarities)
print(f"Result: {corpus[top_idx]}")
```


</div>

</div>

---

# Challenges and Limitations

<div class="grid grid-cols-2 gap-4 mt-3">

<div class="space-y-3">

## Technical Challenges

<div class="p-3 bg-red-50 border border-red-200 rounded">
<h4 class="font-bold text-red-700 text-sm">Computational Complexity</h4>
<ul class="mt-2 text-sm">
<li>Quadratic attention complexity O(n²)</li>
<li>Memory requirements</li>
<li>Training resources</li>
</ul>
</div>

<div class="p-3 bg-orange-50 border border-orange-200 rounded">
<h4 class="font-bold text-orange-700 text-sm">Representation Quality</h4>
<ul class="mt-2 text-sm">
<li>Loss of fine-grained information</li>
<li>Fixed-size bottleneck</li>
</ul>
</div>

</div>

<div class="space-y-3">

## Practical Issues

<div class="p-3 bg-blue-50 border border-blue-200 rounded">
<h4 class="font-bold text-blue-700 text-sm">Domain Adaptation</h4>
<ul class="mt-2 text-sm">
<li>General models vs. domain-specific</li>
<li>Transfer learning challenges</li>
</ul>
</div>

<div class="p-3 bg-purple-50 border border-purple-200 rounded">
<h4 class="font-bold text-purple-700 text-sm">Evaluation Metrics</h4>
<ul class="mt-2 text-sm">
<li>Semantic Textual Similarity (STS)</li>
<li>Downstream task performance</li>
</ul>
</div>

</div>

</div>

<div class="mt-4 p-3 bg-gray-50 rounded-lg text-center text-sm">
<strong>Key Trade-off:</strong> Expressiveness vs. Efficiency vs. Generalizability
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