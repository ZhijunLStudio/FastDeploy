### **MiniMax-M1 模型架构骨架**

**顶层结构: `MiniMaxM1ForCausalLM`**
这是一个标准的因果语言模型（Causal LM），其主要由两部分构成：

1.  **`MiniMaxM1Model`**: 模型的主体，负责将输入的 Token ID 序列转换为最终的隐藏状态向量序列。
2.  **`LM Head`**: 一个线性投影层，负责将隐藏状态向量映射到词汇表空间，生成预测下一个 Token 的 Logits。

---

#### **骨架核心: `MiniMaxM1Model`**

| 模块 (Module) | 描述 | 关键配置/技术 |
| :--- | :--- | :--- |
| **1. 词嵌入 (Token Embedding)** | 将输入的 Token ID 转换为 `hidden_size` 维度的向量。 | `vocab_size: 200064`<br>`hidden_size: 6144` |
| **2. 循环：80x 解码器层** | 核心计算部分，由 80 个解码器层堆叠而成。每个解码器层处理输入向量并传递给下一层。 | `num_hidden_layers: 80` |
| **3. 最终归一化 (Final Norm)** | 在所有解码器层计算完毕后，对最终的隐藏状态进行一次 RMSNorm。 | `rms_norm_eps: 1e-05` |

---

#### **骨架细节: `MiniMaxM1DecoderLayer` (单个解码器层)**

这是模型最复杂、最核心的部分。一个 Token 在单个解码器层中的数据流如下：

**输入**: `hidden_states`

1.  **输入归一化 (Input Norm)**
    *   **模块**: `RMSNorm`
    *   **操作**: `norm_out = RMSNorm(hidden_states)`
    *   **关键配置**: `rms_norm_eps: 1e-05`

2.  **残差连接点 1**
    *   **操作**: `residual = norm_out` (由于 `postnorm: true`)

3.  **混合自注意力 (Hybrid Self-Attention)** - **[核心分支]**
    *   **决策**: 根据层号 `i` 检查 `attn_type_list[i]` 的值。
    *   **分支 A (`attn_type: 1`) - 标准注意力**:
        *   **模块**: `MiniMaxM1StandardAttention`
        *   **技术**:
            *   **GQA**: 分组查询注意力 (`num_attention_heads: 64`, `num_key_value_heads: 8`)
            *   **Partial RoPE**: 部分旋转位置编码 (`head_dim: 128`, `rotary_dim: 64`, `rope_theta: 10M`)
        *   **操作**: `attn_output = StandardAttention(norm_out)`
    *   **分支 B (`attn_type: 0`) - 线性注意力**:
        *   **模块**: `MiniMaxM1LinearAttention`
        *   **技术**: Mamba/SSM-style 线性时间复杂度注意力（并行扫描算法）。
        *   **操作**: `attn_output = LinearAttention(norm_out)`

4.  **注意力后的 DeepNorm**
    *   **操作**: `hidden_states = residual * beta_attn + attn_output * alpha_attn`
    *   **关键配置**:
        *   `layernorm_full_attention_alpha` / `layernorm_linear_attention_alpha`
        *   `layernorm_full_attention_beta` / `layernorm_linear_attention_beta`

5.  **中间归一化 (Post-Attention Norm)**
    *   **模块**: `RMSNorm`
    *   **操作**: `norm_out = RMSNorm(hidden_states)`
    *   **关键配置**: `rms_norm_eps: 1e-05`

6.  **残差连接点 2**
    *   **操作**: `residual = norm_out`

7.  **MoE + 共享专家 MLP**
    *   **模块**: `MiniMaxM1MoE`
    *   **技术**:
        *   **稀疏 MoE**: 32 选 2 路由 (`num_local_experts: 32`, `num_experts_per_tok: 2`)。
        *   **共享专家**: 有一个共享的 MLP，其输出与 MoE 输出通过 Sigmoid 门控融合 (`shared_moe_mode: "sigmoid"`)。
        *   **激活函数**: SwiGLU (`hidden_act: "silu"`)。
    *   **操作**: `mlp_output = MoE_with_SharedExpert(norm_out)`

8.  **MLP 后的 DeepNorm**
    *   **操作**: `hidden_states = residual * beta_mlp + mlp_output * alpha_mlp`
    *   **关键配置**: `layernorm_mlp_alpha`, `layernorm_mlp_beta`

**输出**: `hidden_states` (传递给下一层)

---

### **架构骨架图 (伪代码)**

```
function MiniMaxM1ForCausalLM(token_ids):
    # 1. Embedding
    hidden_states = Embedding(token_ids)

    # 2. 80 Decoder Layers
    for i in 0..79:
        hidden_states = MiniMaxM1DecoderLayer(hidden_states, layer_index=i)
    
    # 3. Final Norm
    hidden_states = FinalRMSNorm(hidden_states)

    # 4. LM Head
    logits = LM_Head(hidden_states)
    return logits

# ----------------------------------------------

function MiniMaxM1DecoderLayer(hidden_states, layer_index):
    # --- Attention Block ---
    norm_out_1 = RMSNorm_1(hidden_states)
    residual_1 = norm_out_1
    
    if attn_type_list[layer_index] == 1:
        attn_output = StandardAttention(norm_out_1)
        alpha, beta = layernorm_full_attention_alpha, layernorm_full_attention_beta
    else: # attn_type == 0
        attn_output = LinearAttention(norm_out_1)
        alpha, beta = layernorm_linear_attention_alpha, layernorm_linear_attention_beta

    hidden_states = residual_1 * beta + attn_output * alpha

    # --- MLP/MoE Block ---
    norm_out_2 = RMSNorm_2(hidden_states)
    residual_2 = norm_out_2

    moe_output = MoE(norm_out_2) # 32选2
    shared_mlp_output = SharedMLP(norm_out_2)
    gate = Sigmoid(CoefficientLinear(norm_out_2))
    mlp_output = moe_output * (1 - gate) + shared_mlp_output * gate
    
    hidden_states = residual_2 * layernorm_mlp_beta + mlp_output * layernorm_mlp_alpha
    
    return hidden_states
```