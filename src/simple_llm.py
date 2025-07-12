#!/usr/bin/env python3
"""
最简LLM实现 - 用最少代码理解大语言模型核心原理
作者：技术博客示例
功能：实现一个包含注意力机制的简化版Transformer语言模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
random.seed(42)

class SimpleTokenizer:
    """简单的字符级分词器"""
    def __init__(self, text):
        # 获取所有唯一字符并排序，构建词汇表
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        # 字符到索引的映射
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        # 索引到字符的映射
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        """将文本编码为token索引列表"""
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        """将token索引列表解码为文本"""
        return ''.join([self.idx_to_char[i] for i in indices])

class MultiHeadAttention(nn.Module):
    """多头注意力机制 - Transformer的核心组件"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # 线性变换层：将输入投影为Query、Key、Value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 生成Query、Key、Value
        Q = self.q_linear(x)  # (batch_size, seq_len, d_model)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用因果掩码（确保只能看到之前的token）
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        # 应用softmax获得注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重到Value
        attention_output = torch.matmul(attention_weights, V)
        
        # 重塑并通过输出线性层
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.out_linear(attention_output)

class TransformerBlock(nn.Module):
    """Transformer块 - 包含注意力机制和前馈网络"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        # 注意力机制 + 残差连接 + 层归一化
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class SimpleLLM(nn.Module):
    """简化版大语言模型"""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token嵌入层：将token索引转换为向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置嵌入层：为每个位置添加位置信息
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer块堆叠
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4) 
            for _ in range(n_layers)
        ])
        
        # 输出层：将隐藏状态映射到词汇表大小
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 生成位置索引
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token嵌入 + 位置嵌入
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        
        # 通过Transformer块
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 输出投影到词汇表
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
        """生成文本"""
        self.eval()
        
        # 编码输入提示
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids])
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 确保输入长度不超过最大序列长度
                if len(generated_ids) >= self.max_seq_len:
                    input_tensor = torch.tensor([generated_ids[-self.max_seq_len:]])
                else:
                    input_tensor = torch.tensor([generated_ids])
                
                # 前向传播
                logits = self.forward(input_tensor)
                
                # 获取最后一个位置的logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # 应用softmax并采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated_ids.append(next_token)
        
        return tokenizer.decode(generated_ids)

def train_simple_model():
    """训练简单模型的示例"""
    # 准备训练数据（这里使用一个简单的文本）
    text = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    机器学习是人工智能的一个重要分支，它通过算法使计算机能够从数据中学习并做出决策或预测。
    深度学习是机器学习的一个子集，它使用神经网络来模拟人脑的工作方式。
    大语言模型是深度学习在自然语言处理领域的重要应用，能够理解和生成人类语言。
    """
    
    # 初始化分词器和模型
    tokenizer = SimpleTokenizer(text)
    model = SimpleLLM(vocab_size=tokenizer.vocab_size)
    
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 准备训练数据
    input_ids = tokenizer.encode(text)
    
    # 简单的训练循环
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    print("\n开始训练...")
    for epoch in range(100):
        # 随机选择一个序列片段
        start_idx = random.randint(0, max(0, len(input_ids) - model.max_seq_len - 1))
        end_idx = start_idx + model.max_seq_len
        
        # 输入和目标（目标是输入向右移动一位）
        x = torch.tensor([input_ids[start_idx:end_idx]])
        y = torch.tensor([input_ids[start_idx+1:end_idx+1]])
        
        # 前向传播
        logits = model(x)
        
        # 计算损失
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print("训练完成！\n")
    
    # 测试生成
    print("=== 文本生成测试 ===")
    test_prompts = ["人工智能", "机器学习", "深度学习"]
    
    for prompt in test_prompts:
        generated_text = model.generate(tokenizer, prompt, max_new_tokens=30, temperature=0.8)
        print(f"输入: '{prompt}'")
        print(f"生成: {generated_text}")
        print("-" * 50)

if __name__ == "__main__":
    print("=== 最简LLM实现演示 ===")
    print("这是一个教学用的简化版大语言模型实现")
    print("包含了Transformer的核心组件：注意力机制、位置编码、前馈网络等\n")
    
    train_simple_model()

