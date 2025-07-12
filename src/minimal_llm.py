#!/usr/bin/env python3
"""
最简LLM实现 - 核心代码版本
用最少的代码展示LLM的核心原理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # 生成Q, K, V
        Q = self.q_linear(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 因果掩码
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        # 注意力权重和输出
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # 重塑并输出
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_linear(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.feed_forward(x))
        return x

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_len=64):
        super().__init__()
        self.max_len = max_len
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        
        x = self.token_emb(x) + self.pos_emb(pos)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        return self.head(x)
    
    def generate(self, tokenizer, prompt, max_tokens=50):
        self.eval()
        tokens = tokenizer.encode(prompt)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                x = torch.tensor([tokens[-self.max_len:]])
                logits = self.forward(x)
                next_token = torch.multinomial(F.softmax(logits[0, -1], dim=-1), 1).item()
                tokens.append(next_token)
                
        return tokenizer.decode(tokens)

# 使用示例
if __name__ == "__main__":
    # 训练数据
    text = "人工智能是未来科技发展的重要方向。机器学习让计算机能够从数据中学习。深度学习使用神经网络模拟人脑。"
    
    # 初始化
    tokenizer = SimpleTokenizer(text)
    model = SimpleLLM(tokenizer.vocab_size)
    
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 简单训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    tokens = tokenizer.encode(text)
    
    for epoch in range(200):
        x = torch.tensor([tokens[:-1]])
        y = torch.tensor([tokens[1:]])
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 生成测试
    print("\n生成测试:")
    result = model.generate(tokenizer, "人工智能", 20)
    print(result)

