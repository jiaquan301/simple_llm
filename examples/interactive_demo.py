#!/usr/bin/env python3
"""
零基础LLM教程 - 交互式演示代码
这个文件包含了教程中所有的代码，并提供了交互式的演示功能
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
    """
    简单的字符级分词器
    
    这个分词器把文本转换成数字，让计算机能够处理
    就像给每个字符分配一个身份证号码
    """
    def __init__(self, text):
        print("🔤 初始化分词器...")
        # 获取所有唯一字符并排序，构建词汇表
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # 字符到索引的映射（字符 → 数字）
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        # 索引到字符的映射（数字 → 字符）
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f"   词汇表大小: {self.vocab_size}")
        print(f"   包含字符: {self.chars[:10]}..." if len(self.chars) > 10 else f"   包含字符: {self.chars}")
    
    def encode(self, text):
        """将文本编码为token索引列表"""
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        """将token索引列表解码为文本"""
        return ''.join([self.idx_to_char[i] for i in indices])
    
    def demo_encoding(self, text):
        """演示编码过程"""
        print(f"\n📝 编码演示:")
        print(f"   原始文本: '{text}'")
        encoded = self.encode(text)
        print(f"   编码结果: {encoded}")
        decoded = self.decode(encoded)
        print(f"   解码验证: '{decoded}'")
        return encoded

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 - Transformer的核心组件
    
    就像人类阅读时眼睛会在文本中游走，寻找相关信息
    多头注意力让模型能够同时从多个角度理解每个词
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        print(f"🧠 初始化多头注意力:")
        print(f"   模型维度: {d_model}")
        print(f"   注意力头数: {n_heads}")
        print(f"   每个头的维度: {self.head_dim}")
        
        # 线性变换层：将输入投影为Query、Key、Value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 生成Query、Key、Value
        Q = self.q_linear(x)  # 查询：我想找什么信息？
        K = self.k_linear(x)  # 键：每个词能提供什么信息？
        V = self.v_linear(x)  # 值：每个词的具体内容是什么？
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数（相似度）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用因果掩码（确保只能看到之前的token）
        # 这很重要：模型不能"偷看"未来的词
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
    """
    Transformer块 - 包含注意力机制和前馈网络
    
    这是Transformer的基本构建单元，就像搭积木的基本块
    每个块都让模型对输入有更深的理解
    """
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        print(f"🧱 初始化Transformer块:")
        print(f"   模型维度: {d_model}")
        print(f"   前馈网络维度: {d_ff}")
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化：保持数值稳定
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络：先扩展维度（发散思维），再压缩（收敛结论）
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # 激活函数：引入非线性
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        # 注意力机制 + 残差连接 + 层归一化
        # 残差连接：新知识 + 原有知识
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class SimpleLLM(nn.Module):
    """
    简化版大语言模型
    
    这是我们的完整模型，包含了现代LLM的所有核心组件
    虽然很小，但原理和GPT是一样的
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        print(f"\n🤖 初始化SimpleLLM:")
        print(f"   词汇表大小: {vocab_size}")
        print(f"   模型维度: {d_model}")
        print(f"   注意力头数: {n_heads}")
        print(f"   Transformer层数: {n_layers}")
        print(f"   最大序列长度: {max_seq_len}")
        
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
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   总参数数量: {total_params:,}")
        
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
    
    def generate(self, tokenizer, prompt, max_new_tokens=50, temperature=1.0, verbose=False):
        """
        生成文本
        
        参数:
        - prompt: 输入提示
        - max_new_tokens: 最大生成token数
        - temperature: 温度参数（控制随机性）
        - verbose: 是否显示详细过程
        """
        self.eval()
        
        if verbose:
            print(f"\n🎯 开始生成文本:")
            print(f"   输入提示: '{prompt}'")
            print(f"   最大生成长度: {max_new_tokens}")
            print(f"   温度参数: {temperature}")
        
        # 编码输入提示
        input_ids = tokenizer.encode(prompt)
        generated_ids = input_ids.copy()
        
        if verbose:
            print(f"   编码后的输入: {input_ids}")
            print(f"\n📝 生成过程:")
        
        with torch.no_grad():
            for i in range(max_new_tokens):
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
                
                if verbose and i < 10:  # 只显示前10步
                    next_char = tokenizer.decode([next_token])
                    print(f"   步骤 {i+1}: 生成 '{next_char}' (token {next_token})")
        
        result = tokenizer.decode(generated_ids)
        if verbose:
            print(f"\n✅ 生成完成!")
            print(f"   最终结果: '{result}'")
        
        return result

def demonstrate_attention():
    """演示注意力机制的工作原理"""
    print("\n" + "="*60)
    print("🔍 注意力机制演示")
    print("="*60)
    
    print("\n💡 注意力机制就像人类阅读时的眼球运动:")
    print("   当我们读到'他'这个代词时，眼睛会回头寻找'他'指的是谁")
    print("   注意力机制让计算机也能做到这一点")
    
    # 创建一个简单的例子
    d_model = 8
    n_heads = 2
    attention = MultiHeadAttention(d_model, n_heads)
    
    # 模拟输入：3个词，每个词用8维向量表示
    x = torch.randn(1, 3, d_model)
    
    print(f"\n📊 输入形状: {x.shape}")
    print("   (批次大小=1, 序列长度=3, 模型维度=8)")
    
    # 前向传播
    output = attention(x)
    
    print(f"📊 输出形状: {output.shape}")
    print("   注意力机制成功处理了输入!")

def demonstrate_tokenizer():
    """演示分词器的工作原理"""
    print("\n" + "="*60)
    print("🔤 分词器演示")
    print("="*60)
    
    sample_text = "人工智能很有趣"
    tokenizer = SimpleTokenizer(sample_text)
    
    # 演示编码过程
    tokenizer.demo_encoding("人工智能")
    tokenizer.demo_encoding("很有趣")
    
    print(f"\n🔍 词汇表映射:")
    for char, idx in list(tokenizer.char_to_idx.items())[:5]:
        print(f"   '{char}' → {idx}")

def train_and_demo():
    """训练模型并演示生成效果"""
    print("\n" + "="*60)
    print("🎓 模型训练与生成演示")
    print("="*60)
    
    # 准备训练数据
    text = """人工智能是计算机科学的一个分支。机器学习是人工智能的重要组成部分。深度学习使用神经网络来模拟人脑。大语言模型能够理解和生成人类语言。自然语言处理是人工智能的重要应用领域。"""
    
    print(f"📚 训练数据长度: {len(text)} 字符")
    
    # 初始化分词器和模型
    tokenizer = SimpleTokenizer(text)
    model = SimpleLLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2)
    
    # 准备训练数据
    input_ids = tokenizer.encode(text)
    
    # 训练设置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    
    print(f"\n🏋️ 开始训练...")
    losses = []
    
    for epoch in range(200):
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
        
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"   Epoch {epoch:3d}, Loss: {loss.item():.4f}")
    
    print(f"✅ 训练完成! 最终损失: {losses[-1]:.4f}")
    
    # 测试生成
    print(f"\n🎯 文本生成测试:")
    test_prompts = ["人工智能", "机器学习", "深度学习"]
    
    for prompt in test_prompts:
        print(f"\n📝 输入: '{prompt}'")
        generated_text = model.generate(tokenizer, prompt, max_new_tokens=20, temperature=0.8)
        print(f"🤖 生成: {generated_text}")

def interactive_demo():
    """交互式演示"""
    print("\n" + "="*60)
    print("🎮 交互式演示")
    print("="*60)
    
    print("欢迎来到LLM交互式演示!")
    print("你可以选择以下演示:")
    print("1. 分词器演示")
    print("2. 注意力机制演示") 
    print("3. 完整训练和生成演示")
    print("4. 全部演示")
    
    while True:
        try:
            choice = input("\n请选择 (1-4, 或 'q' 退出): ").strip()
            
            if choice == 'q':
                print("👋 再见!")
                break
            elif choice == '1':
                demonstrate_tokenizer()
            elif choice == '2':
                demonstrate_attention()
            elif choice == '3':
                train_and_demo()
            elif choice == '4':
                demonstrate_tokenizer()
                demonstrate_attention()
                train_and_demo()
            else:
                print("❌ 无效选择，请输入 1-4 或 'q'")
                
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    print("🚀 零基础LLM教程 - 交互式演示")
    print("="*60)
    print("这个程序将带你体验LLM的核心组件和工作原理")
    
    # 运行交互式演示
    interactive_demo()

