# AI 领域深度技术专题
> 面向 LLM / Agent / RAG 方向，结合王怡琼个人背景深度定制

---

## 目录

1. [专题一：Advanced RAG（生产级 RAG 深度原理）](#专题一)
2. [专题二：LLM 训练与对齐（RLHF / DPO / KTO 深层原理）](#专题二)
3. [专题三：LLM 推理优化（vLLM / PagedAttention / 显存优化）](#专题三)
4. [专题四：MoE / 长上下文 / 模型架构演进](#专题四)
5. [专题五：企业级 AI Agent 开发——江森自控实战深度解析](#专题五)
6. [专题六：你的多项目对比与整合叙事](#专题六)
7. [附录：面试核心知识点速查卡](#附录)

---

<a id="专题一"></a>
## 专题一：Advanced RAG（生产级 RAG 深度原理）

### 1.1 RAG 演进三阶段

```
Naive RAG（基础版）:
  Query → Retrieve → Generate（单轮）

Advanced RAG（你用的，港口+物流枢纽项目）:
  Query → Pre-Retrieve（Query改写）→ Retrieve
        → Post-Retrieve（重排序）→ Generate

Modular RAG（未来方向）:
  Query → [Router] → [Search/Deduce/Memory] → [Evaluate] → Generate
```

### 1.2 Pre-Retrieval：Query 改写（结合你的项目）

**你的物流枢纽项目为什么需要 Query 改写？**
> "政策咨询场景的 Query 改写很关键。用户的问题往往是口语化的，比如'今年枢纽评价多了哪些指标'，但知识库里写的是'根据《关于开展2025年度国家物流枢纽建设工作的通知》，新增指标包括……'。表述差异很大，直接检索效果差。"

**具体方案：**
- **Query Decomposition**：把复杂问题拆成多个子问题分别检索
- **HyDE（假设文档嵌入）**：用 LLM 生成假设答案，用假设答案去检索
- **Query Expansion**：用 LLM 把单问题扩展为多角度 query 集

### 1.3 Post-Retrieval：上下文压缩与过滤

**Context Compression（你的项目中如何做）：**
> "两个项目我都做了上下文压缩，但不是简单截断。港口项目的方案是用 CrossEncoder 打分，过滤低于阈值的结果；物流枢纽项目的方案更精细——用 LLM 把每个检索块压缩成 2-3 句核心内容，去掉政策文件中的重复表述和废话。实测上下文压缩后，相同 token 数下有效信息密度提升了约 40%。"

### 1.4 Self-RAG：自适应检索增强（下一代方向）

**Self-RAG 核心机制（斯坦福 2024 年论文）：**
> "Self-RAG 的创新是让模型自己决定什么时候检索。模型在生成过程中会输出特殊的 [Retrieval] token，遇到需要查资料的地方就停下来。检索后，模型还会输出 [Is Supported] 或 [Contradict] 来自我校验。这比全程无脑检索更高效，也能自我纠正幻觉。"

```
Self-RAG 生成示例：
"根据《关于开展2025年度...的通知》[Retrieval]
"新增指标包括智能仓储... [Relevant]"[检索结果]
"该政策要求枢纽配备... [Is Supported]"[模型自我校验]
"因此，该枢纽符合... [Generation]"[最终输出]
```

### 1.5 RAG 评估体系（RAGAS + 你的自定义指标）

| 指标 | 含义 | 你在项目中的数值 |
|------|------|----------------|
| Faithfulness | 忠实于检索上下文 | 提升 18% |
| Answer Relevancy | 回答与问题相关 | — |
| Context Precision | top-k 排序质量 | CrossEncoder 提升 15% |
| 事实准确率（自定义）| LLM Judge 判断 | 62%→89% |
| 术语覆盖率（自定义）| 专业术语使用率 | — |

---

<a id="专题二"></a>
## 专题二：LLM 训练与对齐（深层原理）

### 2.1 RLHF 三阶段完整流程

```
Stage 1: SFT
  人类编写标准答案 → 监督学习
  问题：高质量数据成本高（GPT-4 每条约 5-10 元）

Stage 2: Reward Model
  人类偏好排序（同一问题的 A vs B 哪个更好）
    ↓
  训练 Reward Model（Bradley-Terry 模型）
  L(θ) = -E[log σ(r_θ(x,y_c) - r_θ(x,y_r))]

Stage 3: PPO 优化
  用 RM 信号优化主模型，KL 约束防止偏离 SFT 太远
  L = E[奖励] - β · KL(π_θ / π_ref)
```

**面试口述版（结合你的经验）：**
> "RLHF 训练最大的问题是 Reward Hacking——模型学会'欺骗' RM，而不是真正提升回答质量。比如 RM 可能会被模型发现'更长的回答总是得分更高'，于是模型开始无意义地堆字数。PPO 的 KL 约束（β 参数）是主要的调节手段：β 太大→模型不敢优化，效果差；β 太小→模型容易跑偏，产生有害输出。"

### 2.2 DPO（Direct Preference Optimization）深度原理

**为什么 DPO 是 2024-2025 主流？**

> "DPO 的核心洞察是：PPO 需要训练 3 个模型（SFT + RM + PPO），训练不稳定，显存占用大。DPO 用数学变换把优化目标变成只需要 2 个模型：SFT reference + 待优化模型。这让训练从 3 阶段简化为 2 阶段，显存减少约 40%，训练速度提升 2-3 倍。"

**DPO 的数学（理解这个能加很多分）：**

```
PPO 的目标函数：
max_π E[r(x,y)] - β · KL[π(y|x) || π_ref(y|x)]

DPO 的等价变换：
→ max_π E_{(x, y_c, y_r)}[log σ(β · (log π(y_c|x) - log π_ref(y_c|x)
                                              - log π(y_r|x) + log π_ref(y_r|x)))]

直观理解：让模型增大 chosen 回答的相对概率，减小 rejected 回答的相对概率
关键：直接用策略模型自己的概率，不需要单独的 reward model！
```

**你如何在实际项目中使用 DPO：**
> "我的两个项目目前主要用 SFT（LoRA）——因为我们缺乏大规模偏好数据标注能力（标注一对 chosen/rejected 成本 15-30 元，需要 1-2 万对才够）。DPO 的方向我知道是正确的，未来我会构建偏好数据集，对齐质量可以进一步提升。"

### 2.3 对齐数据标注 pipeline（工程实践）

```
Step 1: SFT 数据生成
  - GPT-4 生成多样化问答对（few-shot prompting）
  - 人工校验（标注成本：约 5-10 元/条）

Step 2: 偏好数据标注（关键！）
  - 同一问题，两个模型的答案
  - 标注员判断：A vs B 哪个好？（Elo 排序）
  - 质量控制：每样本 3 人标注，Kappa > 0.7
  - 成本：约 15-30 元/对

Step 3: DPO 训练
  - 使用 LLaMA-Factory
  - 8×A100 训练 7B 模型约 4-8 小时
```

---

<a id="专题三"></a>
## 专题三：LLM 推理优化（vLLM / PagedAttention / 显存优化）

> 这是你 RTX 5090 项目最相关的深度知识——面试官很喜欢问你"8-bit 量化怎么做的"。

### 3.1 LLM 推理的两个阶段

```
Prefill（编码阶段）：
  输入 Prompt → 计算 KV Cache → 生成第一个 Token
  特点：计算密集，GPU 利用率高
  时间 ∝ len(prompt) × d² / GPU算力

Decode（解码阶段）：
  逐 Token 自回归生成
  特点：memory-bound（内存带宽瓶颈），串行程度高
  时间 ∝ len(response) × d² / GPU带宽

⚠️ Decode 是推理延迟的主要瓶颈，也是优化的主战场！
```

### 3.2 KV Cache 的显存问题

**问题（结合你的 RTX 5090 项目讲）：**
> "每个 token 都会产生 KV 向量，存储在 GPU 显存中。Qwen2.5-14B，上下文 4096 tokens 时，KV Cache 约 160GB——这超过了单卡 A100 的 80GB。我的 RTX 5090 项目通过 8-bit 量化，把这个占用减半到约 80GB，才得以在单卡上运行。"

```
KV Cache 显存 = 2 × layers × heads × dim_per_head × seq_len × bytes_per_param
             = 2 × 80 × 8 × 128 × 4096 × 2 bytes（FP16）
             ≈ 160 GB（14B 模型，4096 tokens）
```

### 3.3 PagedAttention（vLLM 核心创新）

**核心思想（类比 OS 虚拟内存）：**
> "标准 KV Cache 按连续方式存储，但生成时 Token 数量动态变化，会产生大量显存碎片（就像内存碎片一样）。PagedAttention 把 KV Cache 划分成固定大小的'页'（类比 OS 页表），每个序列不需要连续显存，可离散存储。这将显存碎片率从 60-80% 降到 4% 以下，吞吐提升 24 倍！"

```
传统（连续分配）：
Seq1: [KV1][KV2][KV3][KV4][空][空] ← 碎片
Seq2: [KV1][KV2][空][空][空][空]   ← 碎片

PagedAttention（分页管理）：
Block 0: [Seq1_KV1][Seq1_KV2]
Block 1: [Seq1_KV3][Seq1_KV4]
Block 2: [Seq2_KV1][Seq2_KV2]
Block 3: [空][空] ← 按需分配，无碎片
```

### 3.4 你的 8-bit 量化实战经验（面试加分项）

**INT8 量化怎么做？**
> "8-bit 量化有两种主流方案：① 后训练量化（PTQ）——训练后直接量化，不需要重新训练，数据依赖低；② 量化感知训练（QAT）——训练时模拟量化，效果好但需要重新训练。我的项目用后训练量化（GPTQ/AWQ），因为快且不需要重训。"

**NF4（4-bit NormalFloat）量化（QLoRA 核心技术）：**
> "NF4 的设计基于 LLM 权重通常是呈现均匀分布的假设。用 4-bit 表示 16 个层级，比 INT4 的随机量化更准确。QLoRA 把 NF4 用于原模型权重（保持高精度训练），LoRA 部分保持 FP16。这种'混合精度'策略让我在 RTX 5090 上训 14B 模型：原模型 4-bit NF4（~7GB）+ LoRA FP16（~0.8GB）+ KV Cache ~70GB ≈ 单卡可跑。"

**异常值处理（你踩过的坑）：**
> "量化时有个关键问题：LLM 权重有少数异常大值（outliers），直接量化会损失大。我用混合精度方案：异常值维度保持 FP16，正常维度 NF4，混合后总精度损失 <1%。"

### 3.5 Continuous Batching（持续批处理）

> "GPU 处理请求时会等待所有 batch 序列生成完（EOS token），然后才能处理下一批。Continuous Batching 的改进是：每生成一个 token 就检查是否有序列结束，一旦结束立即调度新请求进来。这样 GPU 始终高利用率，吞吐提升 3-5 倍。vLLM 默认开启此功能。"

---

<a id="专题四"></a>
## 专题四：MoE / 长上下文 / 模型架构演进

### 4.1 MoE（Mixture of Experts）

**MoE 核心思想：**
> "不是所有参数都对每个 token 生效，而是'专家分工'——每个 token 只激活少数专家网络（top-2 或 top-1）。总参数量大，但实际计算量只和激活的专家数量相关。"

```
标准 FFN：y = FFN(x)  # 所有参数都参与

MoE FFN：
  gate(x) = Softmax(W_g × x)
  top_k = TopK(gate(x), k=2)     # 只激活 top-2 专家
  y = Σ_i top_k[i] * FFN_i(x)   # 加权求和

效果：157B 参数，1.1B FLOPs/token（和 7B Dense 模型计算量相当）
```

### 4.2 DeepSeek MoE 的两个创新（国产骄傲，必考）

**① Fine-grained Expert Segmentation（细粒度专家分割）：**
> "传统 MoE 把 FFN 切成 8 个专家，粒度粗，负载不均衡。DeepSeek 把每个专家再细分成多个子专家，增加专业化的同时保持总参数量。这让专家利用率从 30-40% 提升到接近 100%。"

**② Shared Expert（共享专家）：**
> "所有 token 都经过公共专家（不是 Router 决定），捕获通用知识（如语法、常识）。减少重复——否则每个专家都学一遍通用知识，浪费容量。"

### 4.3 RoPE 位置编码与长上下文

**为什么长上下文难？**

| 挑战 | 原因 | 解决方案 |
|------|------|---------|
| Attention 二次复杂度 | O(n²) | Flash Attention（O(n)）|
| 位置编码外推 | 训练最长 4K，推理 128K | RoPE θ 调整 / YaRN |
| KV Cache 显存 | 128K tokens 显存巨大 | PagedAttention |
| 远距离信息衰减 | Attention 对远距离弱 | LongRoPE / LM-Infinite |

**RoPE 外推原理（面试高阶问题）：**
> "RoPE 把绝对位置编码成旋转矩阵角度。训练时见过 position 1~4096，推理用到 128K。外推方案：① 线性外推（直接扩展，但高频维度效果差）；② RoPE θ 缩小（YaRN 方案，减小 base θ 让高频衰减更慢）；③ 你的项目中怎么处理的？——我的物流枢纽项目主要是 8K 以内的上下文，暂时没遇到外推问题，但如果更长，可以用 LongRoPE 做微调。"

---

<a id="专题五"></a>
## 专题五：企业级 AI Agent 开发——江森自控实战深度解析

> 这是你最独特的经历——世界 500 强企业里的 AI Agent 商业落地，你的差异化核心优势。

### 5.1 企业级 AI Agent 的架构特点

**vs 个人项目/科研项目的核心区别：**

```
个人/科研项目：
  输入 → Prompt → LLM → 输出（确定性低，可接受错误）

企业级 Agent（江森自控）：
  用户查询 → 输入校验 → 意图分类 → Schema匹配 → CoT推理
          → Tool Calling → 执行验证 → 结果审计 → 输出
  （确定性高，每步必须有质量保证和日志记录）
```

**企业级 Agent 的 4 个核心要求：**

| 要求 | 你在江森自控的做法 |
|------|-----------------|
| **可审计性** | 每条 SQL 执行有日志记录，可追溯到原始数据 |
| **确定性** | CoT 链路固定，输出格式标准化，减少随机性 |
| **稳定性** | Tool 执行失败有 fallback，不崩溃 |
| **准确性** | 执行前后都做业务逻辑校验，SQL 执行成功率是核心指标 |

### 5.2 CoT 推理链路设计（你的核心贡献）

**你的 4-6 步 CoT 框架：**

```
Step 1: 意图理解（Intent Classification）
  输入：用户自然语言查询
  输出：查询类型（数值查询/对比查询/趋势查询/汇总查询）
  Prompt 策略：Few-shot 示例覆盖 4 种类型

Step 2: Schema 匹配（Schema Matching）
  输入：查询类型 + 数据目录
  输出：涉及的数据表和字段
  你做的：Schema 描述注入——每个表的字段都有详细描述和示例值

Step 3: SQL 生成（SQL Generation）
  输入：Schema + 用户问题 + Few-shot 示例
  输出：SQL 查询语句
  你做的：Prompt 模板化，包含输出格式约束和边界条件说明

Step 4: 语法校验（Syntax Validation）
  自动检查 SQL 语法正确性，避免空执行

Step 5: 执行与结果验证（Execution + Validation）
  执行 SQL → 检查结果集（空结果？异常值？）→ 如异常则回退

Step 6: 自然语言生成（NLG）
  把数据结果转化为用户可理解的分析结论
```

### 5.3 Prompt Engineering 实战经验（你的核心技能）

**Prompt 三层结构（你设计的）：**

```
┌─────────────────────────────────────┐
│ Layer 1: System Prompt（系统级，全局不变）│
│ - 角色定义：你是一个专业的财务数据分析助手  │
│ - 输出格式：必须包含数字、百分比、对比信息  │
│ - 安全约束：不能直接暴露原始金额，可模糊化  │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Layer 2: Task Prompt（任务级，每轮更新）│
│ - 当前问题 + 上下文（前几轮对话摘要）     │
│ - 相关 Schema 信息（当前任务涉及的表）  │
│ - CoT 指令（请按以下步骤推理...）       │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Layer 3: Few-shot Examples（示例级）  │
│ - 3-5 个典型问答示例                  │
│ - 覆盖各种查询类型和边界情况            │
└─────────────────────────────────────┘
```

**你遇到的典型问题及解决：**

> **问题：Agent 生成的 SQL 字段名总是匹配错误**
> → 解决：Schema 注入 + 相似字段名列表 + 校验步骤

> **问题：Agent 在边界情况乱编数字**
> → 解决：在 Prompt 里加"如果数据不存在，请明确说不知道，不要编造"

> **问题：CoT 步数太多累积误差**
> → 解决：4-6 步最优，每步加结果验证点，不正确则回退重试

### 5.4 Tool-use 接口设计（微软 AI 框架）

**你的 Tool 定义结构：**

```json
{
  "name": "execute_financial_sql",
  "description": "执行财务数据 SQL 查询，返回结构化结果集。
                   适用于：区域对比/时间趋势/指标计算类问题。
                   不适用于：开放分析/需要外部数据的问题。",
  "parameters": {
    "type": "object",
    "properties": {
      "sql_query": {
        "type": "string",
        "description": "标准 SQL 查询语句，必须包含具体的字段和过滤条件。
                       示例：SELECT region, SUM(revenue) FROM fact_sales
                            WHERE fiscal_quarter = 'Q3_2025' GROUP BY region"
      },
      "include_metadata": {
        "type": "boolean",
        "description": "是否返回元数据（执行时间、影响行数）用于调试"
      }
    },
    "required": ["sql_query"]
  }
}
```

**Tool 执行结果的 Error Handling：**
```python
# 你设计的错误处理策略
result = execute_sql(sql_query)
if result.is_empty:
    return "数据不存在，请尝试调整查询范围或时间区间。"
elif result.error_type == "SYNTAX_ERROR":
    # 回退：简化查询
    simplified = simplify_sql(sql_query)
    return execute_sql(simplified)
elif result.error_type == "TIMEOUT":
    return "查询超时，请减少时间范围或聚合粒度。"
elif result.is_valid:
    return format_result(result)  # 正常格式化输出
```

### 5.5 云原生技术栈（Snowflake + Azure）

**你的技术栈（面试要能讲清楚）：**
```
Snowflake（数据仓库）：
  - 企业级 SaaS 数据仓库，支持 PB 级数据
  - 你用 SQL 从 Snowflake 查询财务数据
  - 优势：自动扩缩容，不用管基础设施

Azure Data Factory (ADF)（数据集成）：
  - 你用 ADF 做 ETL：Snowflake → Data Lake → Power BI
  - Ingestion（数据接入）→ Transformation（清洗转换）→ Load（入库）
  - ADF Pipelines 支持调度和监控

Power BI（可视化）：
  - 你负责 SOR → Power BI 的 ETL 映射规则
  - 交付跨境财务数据的可视化报表
```

### 5.6 DevOps 与 CR 管理（你掌握的工程方法论）

**CR（Change Request）全生命周期管理：**

```
需求提出 → 技术评估（FS评审）→ 开发 → SIT测试 → UAT → 部署上线
   ↑           ↑
  业务方      你写的 FS（功能规格说明书）
```

**你主导的 4 项 CR 交付经验（面试重点）：**
> "我作为 ITBA，主导了 4 项业务变更（CR）的全生命周期管理。具体包括：① 需求分析——和业务方对齐需求，输出 SOR（系统需求规格说明书）和 FS（功能规格说明书）；② ETL 设计——完成从 Snowflake 到 Power BI 的数据映射规则设计；③ SIT 测试——用 SQL 逻辑校验确保数据准确性（跨境财务数据 100% 通过）；④ 上线支持——配合运维团队完成生产环境部署。"

---

<a id="专题六"></a>
## 专题六：你的多项目对比与整合叙事

### 6.1 三个项目的差异化定位

| 维度 | 港口教育大模型 | 物流枢纽大模型 | 江森自控 BA |
|------|------------|------------|---------|
| 场景 | 教育/教学 | 政务/申报 | 财务/商业 |
| 核心创新 | 分数增强算法 | 政策溯源体系 | CoT 推理链路 |
| 模型 | Qwen-14B | Qwen2.5-14B | 微软 AI 框架 |
| 优化 | LoRA | LoRA + 8-bit 量化 | Prompt + Tool-use |
| 检索 | 三路混合检索 | 双阶段 RAG | — |
| 部署 | FastAPI | FastAPI + Docker | 企业内部平台 |
| 规模 | 200+ 学生用户 | 全栈自动化 | 全球业务团队 |

### 6.2 一条主线的叙事方式

**你的 AI 能力成长故事：**

> "我的 AI 能力发展有一条清晰的主线：
>
> **第一阶段（港口教育大模型）**——我入门了 RAG 全链路，核心创新是分数增强算法，理解了检索和生成的关系。
>
> **第二阶段（物流枢纽大模型）**——我在 Qwen2.5-14B 上做了完整的 LoRA + 8-bit 量化 + FastAPI 部署工程，掌握了从模型训练到用户可用的完整工程闭环。
>
> **第三阶段（江森自控实习）**——我在世界 500 强企业里真正落地了 AI Agent，理解了 CoT 推理、Prompt Engineering 和 Tool-use 的工程实践，特别是'确定性'和'可审计性'这些企业级要求。
>
> 现在我正在补齐的：推理优化（vLLM / PagedAttention）、Multi-Agent 协作（项目 3），以及 RLHF 对齐方法的实战经验。"

### 6.3 面试官视角：他们最想看到什么

**算法研究员：** 你有真实创新（分数增强算法），不是纯粹调参；你理解底层原理（LoRA 的低秩假设、CoT vs ReAct 的适用场景）

**工程负责人：** 你能交付端到端系统（物流枢纽项目 FastAPI + Docker）；你在企业环境有实战经验（江森自控 4 项 CR 交付）

**产品/业务负责人：** 你有业务理解能力（财务 Agent、政务申报都是真实业务）；你有降本增效意识（物流枢纽项目把 3 天→30 分钟）

---

<a id="附录"></a>
## 附录：面试核心知识点速查卡

```
【模型架构】
✓ Self-Attention O(n²) → Flash Attention O(n) 的原理
✓ RoPE 位置编码的旋转矩阵实现
✓ GQA / MQA 与 KV Cache 显存优化的关系
✓ MoE 的 expert routing 和 DeepSeek 的 Shared Expert 创新

【训练与对齐】
✓ RLHF 三阶段目的：SFT（学回答）→ RM（学偏好）→ PPO（优化策略）
✓ DPO 为什么省显存：不需要 RM，直接优化策略
✓ DPO vs KTO：DPO 需要成对数据，KTO 只需要判断单条好坏

【推理优化】（你的 RTX 5090 经验相关）
✓ PagedAttention 核心：类比 OS 页表，离散存储 KV Cache
✓ NF4 量化的设计：均匀分布假设，混合精度避免异常值损失
✓ Continuous Batching：EOS 结束立即调度新请求，吞吐提升 3-5x
✓ 你的实际经验：QLoRA + NF4 + 混合精度 → 单卡跑 14B 模型

【RAG】
✓ HyDE：用假设答案检索，解决语义 Gap
✓ Context Compression：LLM 压缩每块到 2-3 核心句
✓ Self-RAG：[Retrieval] + [Is Supported] token 机制
✓ 你的项目：CrossEncoder 重排序 + 置信度阈值过滤

【Agent】（你的江森自控核心经验）
✓ CoT vs ReAct：CoT 纯推理适合固定数据源，ReAct 适合开放世界
✓ Tool-use 接口三要素：Description + Schema + Error Handling
✓ CoT 链路最佳步数：4-6 步，每步有验证点
✓ Prompt 三层：System → Task → Few-shot

【评估】
✓ RAGAS 四指标：Faithfulness / Relevancy / Precision / Context
✓ 你的自定义指标：事实准确率（LLM Judge）+ 术语覆盖率
```

---

_本文档与《AI面试准备指南》配套，结合王怡琼个人经历（江森自控 BA 实习 + 物流枢纽大模型 + 智慧港口教育大模型）深度定制。建议配合面试指南一起复习。_
