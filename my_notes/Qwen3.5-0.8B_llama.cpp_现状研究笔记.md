# Qwen3.5-0.8B 在 llama.cpp 中的现状研究笔记

> 基于当前工作区内 `llama.cpp` 源码（`a6cc43c28`）、`Qwen3.5_llama.cpp_研究规划.md` 与 `Qwen3.5-0.8B中的VL部分.docx` 整理。

## 1. 先给结论

当前这份 `llama.cpp` 代码里，`Qwen3.5-0.8B` 的 **decoder 主干支持是明确存在的**，而且不是“占位”支持，而是已经把两条核心路径都建出来了：

- `full attention` 路径已经实现在 `src/models/qwen35.cpp`
- `linear attention / GatedDeltaNet` 路径也已经实现在 `src/models/qwen35.cpp`
- 混合层调度依赖 `full_attention_interval`，默认每 4 层 1 个 full attention，其余层是 recurrent / linear attention

但如果研究对象是 **Qwen3.5-0.8B VL**，那么需要特别注意：

- `llama.cpp` 里并没有单独的 `qwen35vl.cpp`
- 当前实现更像是：
  - 文本解码器走 `QWEN35`
  - 视觉编码器 / mmproj 走 `QWEN3VL` 风格实现
- `libmtmd` 当前 **不支持 video input**
- 因此你手里 `.docx` 中“图像/视频统一路径”的描述，和当前 `llama.cpp` 的可运行现状之间是有差距的

## 2. 架构映射关系

### 2.1 文本主干

- `Qwen3_5ForConditionalGeneration` / `Qwen3_5ForCausalLM` 在转换脚本里都映射到 `QWEN35`
- 入口类是 `Qwen3_5TextModel`
- 图构建入口是 `llm_build_qwen35`

对应位置：

- `llama.cpp/convert_hf_to_gguf.py`
- `llama.cpp/src/models/qwen35.cpp`
- `llama.cpp/src/llama-model.cpp`

### 2.2 视觉侧

- `Qwen3_5ForConditionalGeneration` 在 `--mmproj` 转换模式下，会复用 `Qwen3VLVisionModel`
- 视觉 projector type 被写成 `QWEN3VL`
- 视觉 graph builder 是 `tools/mtmd/models/qwen3vl.cpp`

这意味着当前仓库对 `Qwen3.5-VL` 的理解不是“再造一套独立视觉后端”，而是“Qwen3.5 文本主干 + Qwen3VL 风格 mmproj”

## 3. Decoder 的真实执行路径

### 3.1 层类型判定

`llama.cpp/src/llama-model.cpp` 会读取：

- `ssm_d_conv`
- `ssm_d_inner`
- `ssm_d_state`
- `ssm_dt_rank`
- `ssm_n_group`
- `full_attention_interval`

然后按下面规则标记 recurrent 层：

```cpp
hparams.recurrent_layer_arr[i] = ((i + 1) % full_attn_interval != 0);
```

如果 `full_attention_interval = 4`，则：

- 第 3, 7, 11, 15, 19, 23 层是 full attention
- 其余 18 层是 linear attention

这和你规划文档里“24 层中 18 层线性 + 6 层全注意力”的判断一致。

### 3.2 顶层前向流程

`llm_build_qwen35` 的主循环逻辑很清楚：

1. `build_inp_embd()` 做 token embedding lookup
2. 每层先做 `attn_norm`
3. 如果当前层是 recurrent：
   - 走 `build_layer_attn_linear()`
4. 否则：
   - 走 `build_layer_attn()`
5. attention 输出做 residual
6. 再做 `attn_post_norm`
7. 再做 FFN
8. 再 residual
9. 最后 output norm + lm head

所以从图构建视角，`Qwen3.5` 在 `llama.cpp` 中不是“特殊 case 拼接”，而是一个标准的 hybrid decoder。

## 4. Full Attention 细节

`build_layer_attn()` 里能看到几个关键点：

- `Q` 不是单独投影，而是一次 `wq` 输出 `query + gate`
- `Qcur_full` 被拆成：
  - 真正的 `Qcur`
  - 一个额外 `gate`
- `Q` 和 `K` 都做 RMSNorm
- `Q`/`K` 都走 `ggml_rope_multi(...)`
- 注意力输出之后再乘一个 `sigmoid(gate)`
- 最后再过 `wo`

也就是说，`attn_output_gate=True` 在 `llama.cpp` 中并没有被忽略，而是明确实现成了：

```text
attn_out -> sigmoid(gate) -> elementwise mul -> out_proj
```

这正是你规划里阶段三想确认的一个点。

## 5. Linear Attention / GatedDeltaNet 细节

### 5.1 入口结构

`build_layer_attn_linear()` 里，线性注意力层会构造：

- `qkv_mixed`
- `z`
- `beta`
- `alpha`
- `conv_states`
- `ssm_states`

其中：

- `beta = sigmoid(W_beta x)`
- `alpha_softplus = softplus(W_alpha x + dt_bias)`
- `gate = alpha_softplus * A`

然后会先过一次 `ggml_ssm_conv`，再从结果里切出 `q_conv / k_conv / v_conv`，之后再进入 delta rule。

### 5.2 状态张量

这里的状态不是 KV cache，而是两套 recurrent state：

- `r_l`: rolling conv state
- `s_l`: delta net recurrent state

`llama_hparams` 里这两个大小公式已经写死了：

```text
n_embd_r = (ssm_d_conv - 1) * (ssm_d_inner + 2 * ssm_n_group * ssm_d_state)
n_embd_s = ssm_d_state * ssm_d_inner
```

如果采用你文档里的 0.8B 参数近似代入：

- `ssm_d_conv = 4`
- `ssm_d_state = 128`
- `ssm_n_group = 16`
- `ssm_dt_rank = 16`
- `ssm_d_inner = 16 * 128 = 2048`

则：

- `conv_channels = 2048 + 2 * 16 * 128 = 6144`
- `n_embd_r = 3 * 6144 = 18432`
- `n_embd_s = 128 * 2048 = 262144`

进一步看 `qwen35.cpp` 中的 reshape：

- `conv_states` 被 reshape 成 `[3, 6144, n_seqs]`
- `state` 被 reshape 成 `[128, 128, 16, n_seqs]`

这个结果和你规划里“Conv State 固定大小、SSM state 是大矩阵状态”的判断完全一致，而且现在已经能在 `llama.cpp` 代码里落到具体 shape。

### 5.3 Prefill vs Decode 分叉

`delta-net-base.cpp` 里，`build_delta_net()` 会按 token 数分流：

- `n_seq_tokens == 1`
  - 走 `build_delta_net_autoregressive()`
  - 如果启用 fused，则走 `build_delta_net_fused()`
- `n_seq_tokens > 1`
  - 走 `build_delta_net_chunking()`
  - 如果启用 fused，则走 `build_delta_net_fused()`

所以你规划里的判断是正确的：

- Prefill 不是 KV append 模式，而是 chunked delta rule
- Decode 是真正的 recurrent 单步更新

这也是后续 NPU 迁移里最需要单独画两条路径的地方。

## 6. M-RoPE 在 llama.cpp 中的落地方式

### 6.1 Text / decoder 侧

`QWEN35` 会使用 `rope_sections`

- `llama_model.cpp` 会把 `mrope sections` 打出来
- `qwen35.cpp` 在 full attention 层对 `Q/K` 使用 `ggml_rope_multi`

### 6.2 llama.cpp 内部的位置张量不是 3D，而是 4D 风格

这是一个很容易忽略、但很重要的实现差异：

- 规划和 docx 常用 `[3, B, L]` 讲解 `T/H/W`
- 但 `llama.cpp` 内部对 M-RoPE 输入统一按 `n_pos_per_embd == 4` 处理

对于纯文本 token：

- 第 1/2/3 维位置都复制成同一个标量位置
- 第 4 维写 0

对于图像 embedding：

- `t`
- `y`
- `x`
- `z`

都会占一个槽位

也就是说，研究 `llama.cpp` 的时候，最好把“理论上的 3 维位置”改成“实现上的 4 槽布局”来画图。

## 7. Vision / mmproj 侧的关键实现

### 7.1 Conv3D 在转换时被拆成两个 Conv2D kernel

`convert_hf_to_gguf.py` 里对：

- `visual.patch_embed.proj.weight`

做了显式拆分：

- 第 0 个 temporal slice -> `patch.weight`
- 第 1 个 temporal slice -> `patch.weight.1`

而且直接限制：

- 只支持 `temporal_patch_size == 2`

这和你 docx 中“Conv3D(kernel=(2,16,16)) 可等价看成 patch 级线性映射”的结论是对得上的。

### 7.2 运行时的视觉 graph 不是直接写一个 Conv3D

`tools/mtmd/models/qwen3vl.cpp` 里，视觉前端做的是：

1. 两次 `ggml_conv_2d`
2. 相加
3. reshape / permute
4. 变成 patch token 序列
5. 加 resized learned position embedding
6. 过 vision blocks
7. 2x2 merge
8. 过 projector MLP

所以在 `llama.cpp` 的实际实现里，Conv3D 已经被降解成更适合 ggml 的“两个 2D 卷积 + 维度重排”的等价写法。

### 7.3 Vision encoder 也有独立 RoPE

`qwen3vl.cpp` 的 vision attention 对 `Q/K` 用的是：

- `GGML_ROPE_TYPE_VISION`

这和 decoder 侧的 `M-RoPE` 不是一回事。

因此你的规划文档里这句判断是对的：

- vision encoder 的位置编码
- decoder 侧的 M-RoPE

必须分开研究，不能混为一个算子。

## 8. 当前实现和 docx / 规划之间最重要的偏差

### 8.1 当前 `libmtmd` 不支持视频输入

`tools/mtmd/clip.cpp` 里有明确注释：

- `TODO @ngxson : support both audio and video in the future`

`tools/mtmd/mtmd.cpp` 里在 M-RoPE 位置逻辑处也明确写了：

- `t is omitted as we don't support video input`

所以：

- 你的 docx 中“20 帧视频 -> pixel_values -> vision_encoder”这条链路是理论/Transformers 视角成立
- 但当前本地 `llama.cpp` 现状里，真正能跑通的是 image path，不是完整 video path

### 8.2 当前 mtmd 插入的是“embedding 块”，不是 `image_pad` token 展开

在 `PROJECTOR_TYPE_QWEN3VL` 分支里，`mtmd` 用的是：

- `<|vision_start|>`
- 图像 embedding 块
- `<|vision_end|>`

也就是说当前 `llama.cpp` 的多模态接入点，更接近“直接把视觉 embedding 插到 batch.embd 里”，而不是严格复刻你 docx 里那套 `image_pad/video_pad` 占位符扩展流程。

这不是说 docx 错了，而是：

- docx 更接近 Transformers 处理器视角
- `llama.cpp` 更接近运行时注入 embedding 的工程实现

### 8.3 M-RoPE 图像位置目前是 2D 主导，时间维被弱化

`mtmd_image_tokens_get_decoder_pos()` 当前给图像 token 生成的是：

- `t = pos_0`
- `x = pos_0 + (i % nx)`
- `y = pos_0 + (i / nx)`
- `z = 0`

这说明当前 image path 下：

- 时间维没有真正展开
- 第 4 维目前基本是占位

如果后续你要研究“严格的 T/H/W 三维视觉位置编码如何落到 llama.cpp”，那就必须把“当前 image-only 实现”和“理论上的 video-capable 实现”分开。

## 9. 对研究规划的直接建议

结合当前仓库现状，建议把研究任务拆成两条线：

### 9.1 先做“当前 llama.cpp 已实现链路”

优先顺序建议：

1. `QWEN35` decoder 混合层调度
2. full attention 中的 M-RoPE + output gate
3. GatedDeltaNet 的 prefill / decode 双路径
4. recurrent state 的 shape 与更新点
5. `QWEN3VL` mmproj 图像链路

### 9.2 再做“理论 VL 完整链路和当前实现的 gap”

重点记录：

- video path 缺失
- `image_pad/video_pad` 处理方式差异
- 3 维位置编码理论 vs 4 槽实现
- Conv3D 理论表达 vs ggml 的双 Conv2D 等价实现

## 10. 最值得继续深挖的源码位置

- `llama.cpp/src/models/qwen35.cpp`
- `llama.cpp/src/models/delta-net-base.cpp`
- `llama.cpp/src/llama-memory-recurrent.cpp`
- `llama.cpp/src/llama-hparams.cpp`
- `llama.cpp/tools/mtmd/models/qwen3vl.cpp`
- `llama.cpp/tools/mtmd/mtmd.cpp`
- `llama.cpp/tools/mtmd/mtmd-helper.cpp`
- `llama.cpp/convert_hf_to_gguf.py`

## 11. 一句话总结

如果只问“`Qwen3.5-0.8B` 在 `llama.cpp` 下有没有核心支持”，答案是：**有，而且 decoder 侧已经支持到 GatedDeltaNet 级别。**

如果问“`Qwen3.5-0.8B VL` 是否已经把你 docx 里的完整图像/视频链路一比一落地到 `llama.cpp`”，答案是：**还没有，当前更接近 image-only 的工程化落地，视频链路和一部分 processor 语义还存在 gap。**
