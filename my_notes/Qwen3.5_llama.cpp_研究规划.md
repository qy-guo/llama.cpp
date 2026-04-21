# Qwen3.5-0.8B 在 llama.cpp 内部执行流程深度研究规划

> 2026年04月

---

## 一、任务目标准确重述

本研究任务以 **Qwen3.5-0.8B VL 模型** 为研究对象，围绕其在 llama.cpp 框架内的完整推理执行流程，从「输入 token / 图像/视频」到「输出 logits / 生成文本」进行全链路拆解。研究的核心产出是：对每个执行阶段建立清晰的数据流 + 计算图 + 状态管理认知，并在此过程中明确标注哪些模块、算子、数据结构是未来 NPU 移植时的关键决策点或风险点。

**注意**：当前阶段不需要设计 NPU 方案，但需要带着"这个地方 NPU 能否支持、如何支持"的问题意识去观察和记录。

---

## 二、研究范围

### 在研究范围内

| 范围 | 说明 |
|------|------|
| 模型架构拆解 | Qwen3.5-0.8B 的三大模块：embed_tokens / vision_encoder / decoder |
| llama.cpp 执行路径 | 从 `llama_eval` 到 `ggml_graph_compute` 的完整调用链 |
| 混合注意力机制 | GatedDeltaNet（线性注意力）+ 标准 Full Attention 的计算差异 |
| M-RoPE 位置编码 | T/H/W 三维位置编码的构造与应用方式 |
| 状态管理 | KV Cache（Full Attention）vs Conv State（Linear Attention）的共存机制 |
| 张量形状追踪 | 每个关键节点的 shape 变化 |
| ONNX 模型对照 | 用 ONNX 结构反向验证 llama.cpp 中的计算等价性 |
| Transformers 源码参照 | 用 dev 环境 transformers 源码作为"标准答案"对比 |

### 不在当前研究范围内

- NPU 硬件方案设计、算子映射、内存规划
- llama.cpp 的 GPU 后端（CUDA/Metal）细节
- 量化训练 / QAT
- 生产推理优化（batching、speculative decoding 等）

---

## 三、预备知识：架构快速定位

在开始阶段性研究前，先建立一张架构地图（以 `config.json` 为依据）：

| 模块 | 说明 |
|------|------|
| `embed_tokens` | vocab=248320, dim=1024 |
| `vision_encoder` / PatchEmbed | Conv3D(3→768, kernel/stride=(2,16,16)) |
| `vision_encoder` / 12×VisionBlock | hidden=768, heads=12, Vision-RoPE(H,W only) |
| `vision_encoder` / PatchMerger | 2×2合并 → Linear(3072→3072)+GELU+Linear(3072→1024) |
| `vision_encoder` / 输出 shape | [N_visual_tokens, 1024] |
| decoder Layer 0,1,2 | GatedDeltaNet (linear_attention) — **最特殊，NPU高风险** |
| decoder Layer 3 | Qwen3_5Attention (full_attention) — 标准 GQA |
| decoder 循环规律 | 共 18层线性注意力 + 6层全注意力，每4层一个 full attention |
| Full Attention 参数 | heads=8, kv_heads=2(GQA), head_dim=256, M-RoPE |
| Linear Attention 参数 | key_heads=16, value_heads=16, key/val_dim=128, Conv1D(kernel=4) |
| M-RoPE 参数 | mrope_section=[11,11,10], partial_rotary_factor=0.25, interleaved=True |

---

## 四、阶段性研究路线

### 阶段一：输入预处理与 Embedding 层（1~2天）

#### 关键技术问题

1. llama.cpp 如何将文本 token 和 visual token 合并为统一的 `input_ids` 序列？
2. `<|vision_start|>` / `<|image_pad|>` / `<|video_pad|>` 这些特殊 token 在 llama.cpp 里是如何被识别和处理的？
3. `embed_tokens` 的 lookup 路径在 ggml 计算图里对应哪个节点？
4. Visual token 的 embedding 是在 vision_encoder 之后替换到序列里的，这个替换在 llama.cpp 里发生在哪个位置？

#### 建议验证样例

- **纯文本 case**：单轮对话，`"你好"` 输入 → 追踪从 tokenizer 输出到 embedding 查表的完整路径
- **图像 case**：输入 1 张 224×224 图片 + 短文本，观察 image_pad token 数量是否符合公式：`(H/patch_size/merge_size) × (W/patch_size/merge_size)` = `(224/16/2) × (224/16/2)` = `49`
- **对照验证**：用 `embed_tokens.onnx` 跑同一输入，比较输出 embedding 数值是否一致

#### 重点关注（llama.cpp 侧）

| 位置 | 说明 |
|------|------|
| `llama_tokenize()` | 文本分词，验证特殊 token id（vision_start=248053, image_pad=248056, video_pad=248057）是否正确 |
| `llm_build_inp_embd()` | embedding lookup 的 ggml 实现 |
| `llama_model_load_internal()` 中 vision 模块加载 | 检查 llama.cpp 是否加载了 vision_encoder 权重 |
| `clip_image_preprocess()` 或对应函数 | 图像预处理入口（resize, normalize, patch切分）|

#### 阶段产出

- [ ] **token id 映射表**：特殊 token（vision_start/end/pad, image_pad, video_pad, eos）的 id 值及其在序列里的位置规律
- [ ] **embedding 层数据流图**：text token → lookup → [seq_len, 1024]；visual token → vision_encoder → replace image_pad positions
- [ ] **形状追踪表 v1**：记录 embed_tokens 输出张量 shape

> 🔶 **NPU 留意点**：`embed_tokens` 是一个大型 gather 操作（vocab=248320），NPU 通常对 gather 的支持有限制，需记录词表大小和 embedding 维度。

---

### 阶段二：Vision Encoder 执行流程（2~3天）

#### 关键技术问题

1. 图像/视频从 pixel values 到 visual embedding 的完整计算图是什么？
2. Conv3D patch embedding 的 `kernel=(2,16,16), stride=(2,16,16)` 意味着什么？ggml 里是否有等价实现？
3. 12 层 VisionBlock 的 attention 和 LLM decoder 的 attention 有何不同（特别是 RoPE 类型）？
4. PatchMerger（2×2 merge + projector）的输出 shape 是如何变化的？
5. `pos_embeds`（可学习的位置编码，shape=[2304, 768]）如何处理不同分辨率的图像？

#### 建议验证样例

- **视频 case（核心验证）**：输入 20 帧，300×500 视频
  - 预期中间结果：`pixel_values` shape = `[1, 2880, 1536]`
  - vision encoder 输出：`[1, 720, 1024]`（2880 patches → 720 merged tokens）
  - 用 `vision_encoder.onnx` 计算参考值，与 llama.cpp 对比
- **单图 case**：224×224 图片
  - 预期：T=1（扩展到T=2），patches=(2/2)×(14×14)=196 → merged=49 → shape=`[1,49,1024]`

#### 重点关注（transformers 源码侧）

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `modeling_qwen3_5.py:828` | `Qwen3_5VisionPatchEmbed.forward` | Conv3D等价实现 |
| `modeling_qwen3_5.py:878` | `Qwen3_5VisionAttention.forward` | Vision内的2D RoPE（只有H,W维度，无T）|
| `modeling_qwen3_5.py:848` | `Qwen3_5VisionPatchMerger.forward` | 2×2 merge + projector |
| `modeling_qwen3_5.py:1135` | `Qwen3_5VisionModel.forward` | 整体 vision encoder 入口 |
| `video_processing_qwen3_vl.py` | `Qwen3VLVideoProcessor` | 视频帧采样与预处理 |

#### 阶段产出

- [ ] **Vision Encoder 计算流图**：从 pixel_values 到 image_features 的每一步 shape 变化，标注算子类型
- [ ] **shape 追踪表 v2**：补充 vision encoder 各中间节点的张量 shape
- [ ] **pos_embeds 插值机制说明**：当图像分辨率超出 num_position_embeddings=2304 时如何处理
- [ ] **算子清单**：Conv3D / LayerNorm / VisionAttention(2D RoPE) / GELU / Linear，记录每个算子的参数规模

> 🔶 **NPU 留意点**：Conv3D 在很多 NPU 上支持有限，需确认是否可替换为等价 Linear 实现（ONNX 里已是 Linear）；`pos_embeds` 可学习位置编码的插值是 dynamic shape 操作；Vision Attention 使用 2D RoPE（非 M-RoPE），这是独立的算子。

---

### 阶段三：Decoder 全注意力层（Full Attention）（2~3天）

#### 关键技术问题

1. Qwen3.5 的 Full Attention 使用 GQA（8个Q头，2个KV头），ggml 里如何实现 KV head 的广播？
2. M-RoPE（`mrope_section=[11,11,10]`, `partial_rotary_factor=0.25`, `interleaved=True`）的计算流程是什么？
   - 只有 25% 的 head_dim（即 64 维）参与旋转
   - 64 维按 [11,11,10] 分配给 T/H/W，且是 interleaved 排列
3. `attn_output_gate=True` 是什么机制（Qwen3.5 特有）？
4. Prefill 和 Decode 阶段，KV Cache 的形状和更新机制是什么？

#### 建议验证样例

- **M-RoPE 验证**：取一个包含图像的 prompt，手动计算前 3 个 visual token 的 position_ids（T=0, H=[0,1], W=[0,1]），与 transformers 输出对比
- **GQA 验证**：取 layer 3（第一个 full_attention 层），打印 Q/K/V 的 shape：
  - Q: `[batch, seq_len, 8 heads, 256 dim]`
  - K/V: `[batch, seq_len, 2 heads, 256 dim]`

#### 重点关注（transformers 源码侧）

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `modeling_qwen3_5.py:620` | `Qwen3_5Attention.forward` | GQA + M-RoPE 的完整实现 |
| `modeling_qwen3_5.py:85` | `Qwen3_5TextRotaryEmbedding.forward` | M-RoPE cos/sin 计算 |
| `modeling_qwen3_vl.py:352` | `Qwen3VLTextRotaryEmbedding.forward` | VL 版 M-RoPE（position_ids 是 4D=[4,B,L]）|
| `modeling_qwen3_vl.py:370` | `apply_interleaved_mrope` | interleaved 排列的具体实现 |

#### 阶段产出

- [ ] **M-RoPE 计算流程图**：从 position_ids [3,B,L] → cos/sin [B,L,64] → rotated QK 的每一步
- [ ] **GQA 数据流图**：Q/K/V 的 shape 变化 + head repeat/broadcast 位置
- [ ] **KV Cache 结构说明**：past_key_values 的组织方式，全注意力层的 cache shape
- [ ] **attn_output_gate 机制记录**：对应的权重名称和计算方式

> 🔶 **NPU 留意点**：M-RoPE 是 Qwen3.5 特有的位置编码，`mrope_interleaved=True` 要求特定的维度重排，这个重排算子需要特别关注；GQA 的 KV head broadcast：NPU 是否支持 broadcast attention 或需要显式 repeat？`partial_rotary_factor=0.25`：只旋转前 64 维，后 192 维 pass-through，这是一个 split+concat 操作。

---

### 阶段四：Decoder 线性注意力层（GatedDeltaNet）— 重点阶段（3~4天）

> **这是整个研究中最关键也最特殊的部分，是 Qwen3.5 区别于标准 transformer 的核心，也是 NPU 移植难度最高的部分。**

#### 关键技术问题

1. `Qwen3_5GatedDeltaNet` 的完整计算流程是什么？（不同于 attention，它是 SSM/线性递推结构）
2. **Conv1D 状态（conv_state）** 和 **Full Attention 的 KV Cache** 有何本质区别？
   - conv_state shape 是什么？kernel_size=4 意味着什么？
   - 在 decode 阶段（单步生成）如何更新 conv_state？
3. `chunk_gated_delta_rule` 和 `fused_recurrent_gated_delta_rule` 是什么？哪个用于 prefill，哪个用于 decode？
4. llama.cpp 是否已经实现了 GatedDeltaNet？如果没有，当前是用什么替代或报错的？
5. Prefill（并行模式）vs Decode（递推模式）下，GatedDeltaNet 的计算路径是否有分支？

#### 建议验证样例

- **状态追踪实验**：取 layer 0（第一个 linear_attention 层），输入 5 个 token，打印：
  - conv_state 初始 shape 和更新后 shape
  - 最终输出 hidden_states shape：应为 `[1, 5, 1024]`
- **llama.cpp 支持检验**：尝试将模型加载进 llama.cpp，观察是否有报错或 fallback

#### 重点关注（transformers 源码侧）

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `modeling_qwen3_5.py:356` | `Qwen3_5GatedDeltaNet.__init__` | 理解 conv_dim, key_dim, value_dim 的计算 |
| `modeling_qwen3_5.py:422` | `Qwen3_5GatedDeltaNet.forward` | **最核心**：区分 prefill 和 decode 路径 |
| `modeling_qwen3_5.py:210` | `torch_causal_conv1d_update` | decode 单步的 conv state 更新（CPU fallback实现）|
| `modeling_qwen3_5.py:731` | `Qwen3_5DecoderLayer.forward` | 层级 dispatch：`layer_type == "linear_attention"` 分支 |

#### 阶段产出

- [ ] **GatedDeltaNet 计算流图**：Prefill 路径（chunk_gated_delta_rule）和 Decode 路径（fused_recurrent）的完整数据流
- [ ] **状态管理对比表**：

  | 属性 | Full Attention KV Cache | Linear Attention Conv State |
  |------|------------------------|---------------------------|
  | Shape | [B, kv_heads, L, head_dim] | [B, conv_dim, kernel_size] |
  | 增长方式 | 随序列长度增长 | 固定大小（滑动窗口）|
  | 更新方式 | append | shift+update |
  | 内存规律 | O(L) | O(1) |

- [ ] **llama.cpp 支持情况备忘录**：记录 llama.cpp 当前对该架构的支持状态
- [ ] **算子拆解表**：将 GatedDeltaNet 拆成最小算子单元（Conv1D / matmul / silu / gated delta rule / RMSNorm）

> 🔴 **NPU 留意点（高优先级）**：`fused_recurrent_gated_delta_rule` 是自定义 CUDA kernel，NPU 上无法直接使用，需拆成基础算子重写；Conv1D（depthwise，kernel=4）在 decode 阶段是逐 token 更新的有状态算子；`chunk_gated_delta_rule` 支持分块并行，这可能是 NPU prefill 阶段的实现参考；线性注意力的 decode 模式是**递推**（而非查 cache），NPU 需要两套执行路径。

---

### 阶段五：完整推理流程串联与端到端验证（2~3天）

#### 关键技术问题

1. Prefill 阶段（输入整个 prompt）的完整 ggml 计算图是什么结构？
2. Decode 阶段（每次生成一个 token）的计算图与 Prefill 有何不同？（特别是 linear attention 层）
3. 在 llama.cpp 里，`ggml_graph_compute` 的调度顺序是什么？是否与 transformers 的 forward pass 顺序一致？
4. 一次完整的 VL 推理（图像+文本 → 文字回答）涉及哪些 llama.cpp 函数调用？调用顺序是什么？

#### 建议验证样例

- **端到端数值对齐实验**：
  - 输入：`"这张图片描述了什么？"` + 一张图片
  - 同时运行 transformers（dev 环境）和 llama.cpp
  - 对比每一层输出的 hidden_states 是否数值一致（允许浮点误差 < 1e-3）
  - 如果某层出现偏差，定位是哪个算子导致的
- **Decode 步骤追踪**：生成 5 个 token，每步追踪 linear attention 层的 conv_state 变化

#### 重点关注（llama.cpp 侧）

| 函数/文件 | 说明 |
|----------|------|
| `llama_decode()` | 推理入口，区分 prefill/decode |
| `llm_build_qwen3_vl()` 或对应函数 | VL 版计算图构建 |
| `ggml_graph_compute()` | 最终计算图执行 |
| `llama_sampling_sample()` | 输出 logits 到 token 采样 |

#### 阶段产出

- [ ] **完整执行流程图**：从 `llama_eval()` 调用到 token 输出的函数调用链（含关键 ggml 节点）
- [ ] **Prefill vs Decode 对比图**：两个阶段的计算图差异（重点标注 linear attention 层的不同路径）
- [ ] **数值对齐报告**：列出每层的 cosine similarity 或 max abs error，标注偏差层
- [ ] **端到端时序图**：一次 VL 推理中，vision encoder / embed / decoder 各自占用的时间比例

> 🔶 **NPU 留意点**：Prefill 和 Decode 的计算图不同，NPU 通常需要分别编译两个图（graph compilation）；Vision encoder 只在 prefill 阶段运行一次，是天然适合 NPU 加速的模块；需记录 conv_state 和 kv_cache 的生命周期与 reset 时机。

---

## 五、贯穿全程：NPU 移植关键留意清单

以下内容**当前不需要设计方案**，但在每个阶段都需要主动观察和记录：

| 编号 | 留意点 | 所在阶段 | 风险等级 |
|------|--------|---------|---------|
| N1 | GatedDeltaNet 的 fused CUDA kernel 无法直接移植，必须拆解为基础算子 | 阶段四 | 🔴 高 |
| N2 | Linear Attention 的 decode 路径是有状态递推，与 full attention KV cache 完全不同 | 阶段四 | 🔴 高 |
| N10 | Prefill 和 Decode 必须分别建图，混合层的 dispatch 逻辑需要在 NPU 层面体现 | 阶段五 | 🔴 高 |
| N3 | M-RoPE interleaved 排列要求特定的 gather/scatter 或 reshape 操作 | 阶段三 | 🟡 中 |
| N4 | GQA 的 KV head broadcast（2→8 heads）在 NPU 上可能需要显式 repeat | 阶段三 | 🟡 中 |
| N5 | Conv3D patch embedding 在 NPU 上通常用等价 Linear 替换（ONNX 已验证等价性）| 阶段二 | 🟡 中 |
| N6 | embed_tokens 是大型 gather（vocab=248320），NPU gather 性能需提前评估 | 阶段一 | 🟡 中 |
| N7 | pos_embeds 的动态插值（不同分辨率图像）是 dynamic shape 操作 | 阶段二 | 🟡 中 |
| N8 | partial_rotary_factor=0.25 要求 split+旋转+concat，注意 pass-through 维度 | 阶段三 | 🟢 低 |
| N9 | Vision encoder 和 LLM decoder 的 RoPE 类型不同（2D 纯空间 vs 3D M-RoPE）| 阶段二~三 | 🟢 低 |

---

## 六、资源索引

| 资源 | 路径 |
|------|------|
| Qwen3.5-0.8B 模型文件 | `C:\Users\qingyang.guo\Desktop\conversation\qwen3.5-0.8B\` |
| ONNX 模型（vision_encoder / embed / decoder）| `C:\Users\qingyang.guo\Desktop\conversation\qwen3.5-0.8B-onnx\onnx\` |
| Transformers Qwen3.5 建模源码 | `...\envs\dev\...\transformers\models\qwen3_5\modeling_qwen3_5.py` |
| Transformers Qwen3 VL 建模源码 | `...\envs\dev\...\transformers\models\qwen3_vl\modeling_qwen3_vl.py` |
| VL 预处理源码 | `...\models\qwen3_vl\video_processing_qwen3_vl.py` |
| 已整理的 VL 细节文档 | `C:\Users\qingyang.guo\Desktop\conversation\Qwen3.5-0.8B的VL部分.docx` |
| 模型配置 | `C:\Users\qingyang.guo\Desktop\conversation\qwen3.5-0.8B\config.json` |

---

## 七、各阶段产出汇总

| 阶段 | 核心产出 |
|------|---------|
| 阶段一 | token id 映射表、embedding 数据流图、shape 追踪表 v1 |
| 阶段二 | Vision Encoder 计算流图、算子清单、shape 追踪表 v2 |
| 阶段三 | M-RoPE 计算流程图、GQA 数据流图、KV Cache 结构说明 |
| 阶段四 | GatedDeltaNet 计算流图（Prefill/Decode 双路径）、状态管理对比表、算子拆解表 |
| 阶段五 | 完整执行流程图、Prefill vs Decode 对比图、数值对齐报告、NPU 风险点清单（定稿）|

---

> **核心建议**：建议先跑通阶段四（GatedDeltaNet），因为这是整个架构最陌生、llama.cpp 支持最不确定的部分——如果在 llama.cpp 里无法支持，后面的"端到端验证"目标需要调整到 transformers + ONNX 框架。
