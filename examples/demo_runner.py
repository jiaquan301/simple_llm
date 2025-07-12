#!/usr/bin/env python3
"""
零基础LLM教程 - 非交互式演示代码
直接运行所有演示，无需用户输入
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
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
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
        batch_size, seq_len, d_model = x.shape
        
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.out_linear(attention_output)

class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
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
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4) 
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
        self.eval()
        
        input_ids = tokenizer.encode(prompt)
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if len(generated_ids) >= self.max_seq_len:
                    input_tensor = torch.tensor([generated_ids[-self.max_seq_len:]])
                else:
                    input_tensor = torch.tensor([generated_ids])
                
                logits = self.forward(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated_ids.append(next_token)
        
        return tokenizer.decode(generated_ids)

def demonstrate_tokenizer():
    """演示分词器"""
    print("\n" + "="*60)
    print("🔤 分词器演示")
    print("="*60)
    
    sample_text = "人工智能很有趣"
    tokenizer = SimpleTokenizer(sample_text)
    
    print(f"📚 训练文本: '{sample_text}'")
    print(f"🔍 词汇表大小: {tokenizer.vocab_size}")
    print(f"📝 包含字符: {tokenizer.chars}")
    
    # 演示编码
    test_text = "人工智能"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\n📝 编码演示:")
    print(f"   原始文本: '{test_text}'")
    print(f"   编码结果: {encoded}")
    print(f"   解码验证: '{decoded}'")

def demonstrate_model_training():
    """演示模型训练和生成"""
    print("\n" + "="*60)
    print("🎓 模型训练与生成演示")
    print("="*60)
    
    # 准备训练数据
    text = """人工智能是计算机科学的一个分支。机器学习是人工智能的重要组成部分。深度学习使用神经网络来模拟人脑。大语言模型能够理解和生成人类语言。自然语言处理是人工智能的重要应用领域。"""
    
    print(f"📚 训练数据长度: {len(text)} 字符")
    
    # 初始化分词器和模型
    tokenizer = SimpleTokenizer(text)
    model = SimpleLLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2)
    
    print(f"🤖 模型信息:")
    print(f"   词汇表大小: {tokenizer.vocab_size}")
    print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 准备训练数据
    input_ids = tokenizer.encode(text)
    
    # 训练设置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    
    print(f"\n🏋️ 开始训练...")
    
    for epoch in range(100):
        # 随机选择一个序列片段
        start_idx = random.randint(0, max(0, len(input_ids) - model.max_seq_len - 1))
        end_idx = start_idx + model.max_seq_len
        
        # 输入和目标
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
        
        if epoch % 25 == 0:
            print(f"   Epoch {epoch:3d}, Loss: {loss.item():.4f}")
    
    print(f"✅ 训练完成!")
    
    # 测试生成
    print(f"\n🎯 文本生成测试:")
    test_prompts = ["人工智能", "机器学习", "深度学习"]
    
    for prompt in test_prompts:
        print(f"\n📝 输入: '{prompt}'")
        generated_text = model.generate(tokenizer, prompt, max_new_tokens=15, temperature=0.8)
        print(f"🤖 生成: {generated_text}")

def demonstrate_attention_concept():
    """演示注意力机制概念"""
    print("\n" + "="*60)
    print("🔍 注意力机制概念演示")
    print("="*60)
    
    print("💡 注意力机制的生活例子:")
    print("   想象你在嘈杂的餐厅里和朋友聊天")
    print("   你的大脑会自动过滤掉周围的噪音")
    print("   专注于朋友的声音")
    print("   这就是注意力的作用!")
    
    print("\n🧠 在语言理解中:")
    print("   句子: '小明把书放在桌子上，然后他去了图书馆'")
    print("   当读到'他'时，注意力会回到'小明'")
    print("   因为我们知道'他'指的是小明")
    
    print("\n🔧 计算机如何实现:")
    print("   1. Query (查询): 我想了解什么？")
    print("   2. Key (键): 每个词能提供什么信息？")
    print("   3. Value (值): 每个词的具体内容")
    print("   4. 计算相关性分数")
    print("   5. 根据分数加权组合信息")

def main():
    """主函数 - 运行所有演示"""
    print("🚀 零基础LLM教程 - 完整演示")
    print("="*60)
    print("这个程序将展示LLM的核心组件和工作原理")
    
    # 运行所有演示
    demonstrate_tokenizer()
    demonstrate_attention_concept()
    demonstrate_model_training()
    
    print("\n" + "="*60)
    print("🎉 演示完成!")
    print("="*60)
    print("通过这些演示，你已经了解了:")
    print("✅ 分词器如何将文字转换为数字")
    print("✅ 注意力机制的基本概念")
    print("✅ 完整的LLM训练和生成过程")
    print("\n💡 下一步建议:")
    print("   1. 尝试修改模型参数，观察效果变化")
    print("   2. 使用更多的训练数据")
    print("   3. 学习更高级的LLM技术")
    print("   4. 阅读相关的研究论文")

if __name__ == "__main__":
    main()

