# Minimal LLM Implementation from Scratch

A clean, educational implementation of a Large Language Model (LLM) built from scratch using PyTorch. This project demonstrates the core components of modern language models like GPT, with detailed explanations and minimal dependencies.

## 🎯 Purpose

This implementation is designed for **educational purposes** to help developers and researchers understand:
- How Transformer architecture works
- The mechanics of self-attention and multi-head attention
- Text generation with autoregressive models
- The building blocks of modern LLMs like GPT

📖 **[Read the Complete Tutorial](https://blog.csdn.net/jiaquan3011/article/details/149292522?fromshare=blogdetail&sharetype=blogdetail&sharerId=149292522&sharerefer=PC&sharesource=jiaquan3011&sharefrom=from_link)** - Comprehensive Chinese blog explaining LLM concepts and implementation details.

## 🚀 Features

- **Complete Transformer Implementation**: Multi-head attention, positional encoding, feed-forward networks
- **Two Versions**: Detailed version with extensive comments and minimal version for core understanding
- **Lightweight**: Runs on CPU, no GPU required
- **Educational**: Clear code structure with detailed explanations
- **Runnable**: Works out of the box with minimal setup

## 📁 Files

- `simple_llm.py` - Full implementation with detailed comments (~200 lines)
- `minimal_llm.py` - Streamlined version focusing on core logic (~100 lines)
- `llm_tech_blog.md` - Comprehensive technical blog explaining the concepts (Chinese)

## 🛠️ Requirements

```bash
pip install torch
```

That's it! No other dependencies required.

## 🏃‍♂️ Quick Start

### Run the detailed version:
```bash
python simple_llm.py
```

### Run the minimal version:
```bash
python minimal_llm.py
```

## 📊 Model Architecture

```
SimpleLLM
├── Token Embedding (vocab_size → d_model)
├── Positional Embedding (max_seq_len → d_model)
├── Transformer Blocks (n_layers)
│   ├── Multi-Head Attention
│   │   ├── Query/Key/Value Linear Projections
│   │   ├── Scaled Dot-Product Attention
│   │   └── Causal Masking
│   ├── Layer Normalization
│   ├── Feed-Forward Network
│   └── Residual Connections
└── Output Projection (d_model → vocab_size)
```

## 🔧 Key Components

### 1. Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        # Q, K, V projections and output projection
        
    def forward(self, x):
        # Compute attention scores
        # Apply causal masking
        # Return weighted values
```

### 2. Transformer Block
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        # Attention + Feed-forward + Layer norms
        
    def forward(self, x):
        # Attention with residual connection
        # Feed-forward with residual connection
```

### 3. Text Generation
```python
def generate(self, tokenizer, prompt, max_new_tokens=50):
    # Autoregressive generation
    # Temperature sampling
    # Return generated text
```

## 📈 Example Output

```
=== Minimal LLM Demo ===
Vocab size: 86
Model parameters: 426,838

Training...
Epoch 0, Loss: 4.5614
Epoch 20, Loss: 2.5134
Epoch 40, Loss: 1.4478
...

Text Generation:
Input: 'artificial intelligence'
Output: 'artificial intelligence is the future of technology and machine learning...'
```

## 🎓 Educational Value

This implementation helps you understand:

1. **Attention Mechanism**: How models focus on relevant parts of input
2. **Positional Encoding**: How sequence order is preserved
3. **Autoregressive Generation**: How text is generated token by token
4. **Transformer Architecture**: The building blocks of modern LLMs
5. **Training Process**: How language models learn from data

## 🔍 Code Walkthrough

### Simple Tokenizer
Character-level tokenization for educational purposes:
```python
class SimpleTokenizer:
    def encode(self, text): # text → token IDs
    def decode(self, indices): # token IDs → text
```

### Multi-Head Attention
Core attention mechanism with causal masking:
```python
# Compute Q, K, V
Q, K, V = self.q_linear(x), self.k_linear(x), self.v_linear(x)

# Attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

# Causal mask (prevent looking at future tokens)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))

# Apply attention
attention_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attention_weights, V)
```

## 📚 Learning Path

1. **📖 [Read the Complete Tutorial](https://blog.csdn.net/jiaquan3011/article/details/149292522?fromshare=blogdetail&sharetype=blogdetail&sharerId=149292522&sharerefer=PC&sharesource=jiaquan3011&sharefrom=from_link)** - Start with the comprehensive blog post (Chinese)
2. **Start with `minimal_llm.py`** - Understand the core structure
3. **Study `simple_llm.py`** - Learn detailed implementation
4. **Read the technical blog** - Understand the theory
5. **Experiment with parameters** - See how changes affect performance
6. **Extend the implementation** - Add your own improvements

## ⚡ Performance

- **Model Size**: ~400K parameters
- **Training Time**: < 1 minute on CPU
- **Memory Usage**: < 100MB
- **Inference Speed**: Real-time text generation

## 🔬 Experiments to Try

1. **Change model size**: Increase `d_model`, `n_heads`, `n_layers`
2. **Modify training data**: Use different text datasets
3. **Adjust generation**: Try different temperature values
4. **Add improvements**: Implement RMSNorm, SwiGLU, RoPE

## 🤝 Contributing

This is an educational project. Feel free to:
- Report issues or bugs
- Suggest improvements
- Add more detailed explanations
- Create tutorials or examples

## 📄 License

MIT License - Feel free to use for educational purposes.

## 🙏 Acknowledgments

- Inspired by the "Attention Is All You Need" paper
- Educational approach influenced by Andrej Karpathy's tutorials
- Built for the community to understand LLM fundamentals

## 📞 Contact

For questions about the implementation or suggestions for improvements, please open an issue.

---

**Note**: This is a simplified implementation for educational purposes. For production use, consider established frameworks like Transformers, or more optimized implementations.

