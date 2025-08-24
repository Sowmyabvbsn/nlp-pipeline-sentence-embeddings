---
theme: seriph
background: https://images.pexels.com/photos/6550258/pexels-photo-6550258.jpeg?auto=compress&cs=tinysrgb&w=1920&h=1080
class: 'text-center'
highlighter: shiki
lineNumbers: false
info: |
  ## Distributed Representations Beyond Words and Characters
  Sentence and Paragraph Embeddings in NLP Pipeline
drawings:
  persist: false
css: unocss
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

<div class="grid grid-cols-2 gap-6 mt-6">

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

<div class="mt-6 p-3 bg-blue-50 rounded-lg text-sm">
<strong>Key Challenge:</strong> How do we represent entire sentences and paragraphs as dense vectors while preserving semantic meaning?
</div>

---

# From Words to Sentences: The Compositional Problem

<div class="mt-4">

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

<div class="grid grid-cols-3 gap-3 mt-4 text-xs">

<div class="p-2 bg-red-50 border border-red-200 rounded">
<h4 class="text-red-700 font-bold text-sm">Simple Averaging Issues</h4>
<ul class="mt-1">
<li>Loses word order</li>
<li>Ignores syntax</li>
</ul>
</div>

<div class="p-2 bg-yellow-50 border border-yellow-200 rounded">
<h4 class="text-yellow-700 font-bold text-sm">Bag of Words</h4>
<ul class="mt-1">
<li>"Dog bites man" ≈ "Man bites dog"</li>
<li>Same vectors, different meaning</li>
</ul>
</div>

<div class="p-2 bg-green-50 border border-green-200 rounded">
<h4 class="text-green-700 font-bold text-sm">Solution</h4>
<ul class="mt-1">
<li>Preserve semantics</li>
<li>Consider context</li>
</ul>
</div>

</div>

</div>

---

# What are Sentence Embeddings?

<div class="grid grid-cols-2 gap-6 mt-4">

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

<div class="grid grid-cols-2 gap-4 mt-4">

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

# Popular Sentence Embedding Models

<div class="grid grid-cols-3 gap-3 mt-4">

<div class="p-3 border border-blue-200 bg-blue-50 rounded-lg">
<h3 class="text-blue-800 font-bold text-base mb-2">Universal Sentence Encoder</h3>
<ul class="text-xs space-y-1">
<li><strong>Google's approach</strong></li>
<li>Transformer + Deep Averaging Network</li>
<li>512-dimensional vectors</li>
<li>Multilingual support</li>
</ul>
</div>

<div class="p-3 border border-green-200 bg-green-50 rounded-lg">
<h3 class="text-green-800 font-bold text-base mb-2">Sentence-BERT</h3>
<ul class="text-xs space-y-1">
<li><strong>BERT fine-tuned</strong></li>
<li>Siamese network architecture</li>
<li>Optimized for similarity</li>
<li>Fast inference</li>
</ul>
</div>

<div class="p-3 border border-purple-200 bg-purple-50 rounded-lg">
<h3 class="text-purple-800 font-bold text-base mb-2">InferSent</h3>
<ul class="text-xs space-y-1">
<li><strong>Facebook's model</strong></li>
<li>BiLSTM with attention</li>
<li>Trained on SNLI</li>
<li>4096-dimensional vectors</li>
</ul>
</div>

</div>

<div class="mt-6">

## Performance Comparison

<div class="text-sm">

| Model | Dim | Speed | STS Score | Use Case |
|-------|-----|-------|-----------|----------|
| USE | 512 | Fast | 0.78 | General |
| SBERT | 768 | Very Fast | 0.85 | Similarity |
| InferSent | 4096 | Moderate | 0.75 | Transfer |

</div>

</div>

---

# Paragraph Embeddings: Beyond Sentences

<div class="mt-4">

## Paragraph Embeddings (Doc2Vec)

<div class="grid grid-cols-2 gap-4 mt-4">

<div>

### Key Concepts
- Document-level representations
- Paragraph Vector (PV-DM, PV-DBOW)
- Variable-length handling
- Hierarchical structure

### PV-DM (Distributed Memory)
```text
# Predict word given context + paragraph vector
P(word | context_words, paragraph_id)
```

### PV-DBOW (Distributed Bag of Words)
```text
# Predict words given paragraph vector
P(words | paragraph_id)
```

</div>

<div>

### Modern Approaches

**1. Hierarchical Attention**
- Sentence → Document attention
- Word → Sentence attention

**2. Transformer-based**
- Longformer (long sequences)
- BigBird (sparse attention)
- Hierarchical BERT

**3. Graph-based Methods**
- Sentences as nodes
- Relationships as edges
- Graph Neural Networks

</div>

</div>

</div>

<div class="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm">
<strong>Challenge:</strong> Maintaining coherence across long documents while capturing local and global dependencies.
</div>

---

# Applications and Use Cases

<div class="grid grid-cols-2 gap-6 mt-4">

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

<div class="grid grid-cols-2 gap-4 mt-4">

<div class="space-y-4">

## Technical Challenges

<div class="p-3 bg-red-50 border border-red-200 rounded">
<h4 class="font-bold text-red-700 text-sm">Computational Complexity</h4>
<ul class="text-xs mt-1">
<li>Quadratic attention complexity O(n²)</li>
<li>Memory requirements</li>
<li>Training resources</li>
</ul>
</div>

<div class="p-3 bg-orange-50 border border-orange-200 rounded">
<h4 class="font-bold text-orange-700 text-sm">Representation Quality</h4>
<ul class="text-xs mt-1">
<li>Loss of fine-grained information</li>
<li>Fixed-size bottleneck</li>
</ul>
</div>

</div>

<div class="space-y-4">

## Practical Issues

<div class="p-3 bg-blue-50 border border-blue-200 rounded">
<h4 class="font-bold text-blue-700 text-sm">Domain Adaptation</h4>
<ul class="text-xs mt-1">
<li>General models vs. domain-specific</li>
<li>Transfer learning challenges</li>
</ul>
</div>

<div class="p-3 bg-purple-50 border border-purple-200 rounded">
<h4 class="font-bold text-purple-700 text-sm">Evaluation Metrics</h4>
<ul class="text-xs mt-1">
<li>Semantic Textual Similarity (STS)</li>
<li>Downstream task performance</li>
</ul>
</div>

</div>

</div>

<div class="mt-6 p-3 bg-gray-50 rounded-lg text-center text-sm">
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