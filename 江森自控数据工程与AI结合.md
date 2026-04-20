# 江森自控 ITBA 实习：面试深度准备

> 本文档深度解析你在江森自控的实习经历，帮你找到数据工程与 AI 的结合点，转化为面试优势。

---

## 一、真实情况 vs 面试叙事

### 你的实际情况

> "我主要做数据相关工作：Snowflake 数据仓库维护、ADF ETL pipeline 开发、Power BI 报表、CR 变更管理。没有很密切地做 AI 算法项目。"

### 这是劣势吗？不是！

**AI 应用落地的现实：**
- AI 模型的性能上限，由**数据质量**决定
- RAG 效果差，80% 的问题是**数据 pipeline 没做好**（知识库脏、数据更新不及时、分块不合理）
- Agent 回答不准，往往是**底层数据 Schema 描述不清楚**
- 企业里真正稀缺的是：**既懂数据工程，又懂 AI 应用**的人

**你的差异化定位：**
> "我在江森自控负责数据层的全链路：Snowflake 仓库设计、ADF ETL pipeline、Power BI 报表。我深刻理解企业数据的复杂性，这让我在做 AI Agent 时，能设计出更贴合实际数据的 Schema 和查询逻辑。"

---

## 二、重新理解你在 AI Agent 中的角色

### AI Agent 的数据支撑（你做的事）

```
财务 AI Agent 架构（你支撑的部分）

用户请求
    ↓
[数据层 ← 你做的部分]
Snowflake → ADF Pipeline → 数据清洗 → 语义Schema
    ↓
[LLM/Agent层（你参与的 AI 部分）]
意图理解 → SQL生成 → Tool执行
    ↓
Power BI 报表（你的下游产出）
```

**不要说自己"没做 AI"，你在 AI Agent 中扮演了关键角色：**

### 你做了什么（重新表述）

**① Snowflake 数据仓库维护**
> "我负责企业级 Snowflake 数据仓库的设计与维护，处理来自全球多个业务系统的跨境财务数据。核心挑战是：多源数据（ERP/MES/SCM）的 Schema 映射、数据质量校验、SIT 测试确保 100% 交付准确性。"

→ **AI 关联**：Snowflake 是 AI Agent 的数据来源。Schema 描述的清晰度直接影响 Agent 生成 SQL 的准确率。你对 Schema 的理解，是 Agent 成功的上游保障。

**② ADF ETL Pipeline 开发**
> "我设计并实现了 ADF Pipeline，覆盖 Ingestion → Transformation → Load 全流程。实现了数据从源系统到 Power BI 报表的自动化，支撑了全球财务数据的日级更新。"

→ **AI 关联**：RAG 系统的数据 pipeline 本质上就是 ETL——文档抽取 → 清洗 → 分块 → 向量化 → 入库。你的 ADF 经验完全可以平移到 RAG 数据处理。

**③ Power BI 报表与数据可视化**
> "我负责 SOR（System of Record）到 Power BI 的映射规则设计，将 Snowflake 数据转化为业务可理解的可视化报表。"

→ **AI 关联**：AI Agent 的输出需要呈现给用户，你理解的"如何把数据变成业务可理解的信息"，就是 AI 输出的设计基础。

**④ CR 变更管理与需求文档**
> "我主导了 4 项 CR 的全生命周期管理，撰写 SOR/FS 技术文档，负责需求到交付的端到端把控。"

→ **AI 关联**：AI 产品经理的核心能力——把业务需求转化为技术方案，再转化为 AI 可执行的任务定义。你的 CR 管理经验就是 AI 产品设计的预演。

---

## 三、数据工程 + AI 的 5 个结合点

> 面试时主动提这些，证明你理解 AI 应用落地的全貌。

### 结合点 1：RAG 的数据 Pipeline（你完全可以做）

> "我在江森自控做的 ADF ETL pipeline，和 RAG 系统的数据处理流程本质是一样的：
> - ADF: 原始文档 → 清洗 → 转换 → 入库 Snowflake
> - RAG: 原始文档 → 抽取 → 分块 → 向量化 → 入库 FAISS
> ADF 的数据质量监控理念，也可以平移到 RAG 的数据质量保证。"

### 结合点 2：AI Agent 的数据 Schema 设计

> "AI Agent 生成 SQL 的准确率，很大程度取决于底层数据的 Schema 描述。我在 Snowflake 的工作，让我深刻理解：如何把复杂的企业数据 Schema，翻译成 LLM 能理解的描述。这就是 RAG 里'Context Compression'的核心思想。"

### 结合点 3：Data Mesh + AI Agent 的企业级架构

> "现代企业的 AI Agent 不是孤立的，需要从企业级数据平台获取数据。江森自控的 Snowflake + ADF + Power BI 架构，就是一个简化版的 Data Mesh。AI Agent 是 Data Mesh 的消费层，数据工程师是建设层。"

### 结合点 4：MLOps 与 AI 模型的数据监控

> "AI 模型上线后，数据漂移（Data Drift）是模型失效的主要原因。我在 ADF 中做数据质量监控的经验，可以平移到 MLOps 的模型监控——监控输入数据的分布变化，及时触发模型重训。"

### 结合点 5：从数据到 AI Agent 的端到端思维

> "大多数 AI 研究者只关心模型。但企业里，AI 的价值链条是：数据采集 → ETL → 特征工程 → 模型训练 → 部署 → 监控。我在江森自控完整走通了前半段，这让我的 AI 视角更完整。"

---

## 四、面试话术升级版

### 自我介绍（1分钟版，加入数据工程视角）

> "我叫王怡琼，大连海事大学管理科学与工程硕士，研究方向是基于 Transformer 的碳价时序预测。我有两条并行的能力线：
> **第一条是 AI 技术线**：我有两个 LLM 项目经验——智慧港口教育大模型（RAG+LoRA，准确率+20%）和物流枢纽大模型（Qwen2.5-14B+LoRA+8-bit量化，FastAPI 全栈部署）。
> **第二条是数据工程线**：我在江森自控（Fortune 500）做了 5 个月的 ITBA，负责 Snowflake 数据仓库维护、ADF ETL pipeline 设计和 Power BI 报表开发。
> 我认为 AI 落地的真正瓶颈在数据层，而我的背景正好在数据这层有实战经验，这让我设计的 AI Agent 和 RAG 系统更贴合企业实际。"

### 回答"你在实习中具体做了什么"

**旧版（强调 AI Agent）：**
> "我参与了财务 AI Agent 的 CoT 推理链路设计。"

**新版（强调数据 + AI 的结合）：**
> "我的工作分两部分。**数据层**：我维护企业级 Snowflake 数据仓库，设计了从多源系统到 Power BI 的 ADF ETL pipeline，保证全球财务数据的日级更新和 100% SIT 通过率——这部分奠定了 Agent 能查询到准确数据的基础。**AI 层**：我参与了财务 AI Agent 的 Schema 描述优化，把 Snowflake 里复杂的业务表结构，翻译成 LLM 能理解的语义描述，这直接提升了 Agent 生成 SQL 的准确率。"

### 回答"你没有算法背景，怎么做 AI"

> "我不认为 AI 应用的关键在算法本身。AI 模型的性能，70% 取决于数据质量。我在江森自控的核心工作，就是把数据质量做好、把 Schema 设计清楚、把 Pipeline 打通——这些都是 AI 应用的底层能力。我同时有 AI 技术项目（LLM 微调、RAG 系统）的经验，两条线结合，让我更适合做 AI 应用落地，而不是纯算法研究。"

### 回答"你最大的缺点"

> "我的算法研究深度不如专门做 ML 的人。但我选择了一条更适合自己的路：AI 应用落地。在这个方向，数据的价值往往比算法的边际改进更大。我在江森自控的 5 个月，让我真正理解了企业数据的复杂性，这是纯 AI 研究者很难获得的视角。"

---

## 五、技术细节补充（面试前必掌握）

### 你需要能讲清楚的技术点

**Snowflake：**
- 分层架构：Stage → Raw → Staging → Analytics
- 零拷贝克隆（Zero-Copy Clone）的原理
- Snowflake 在 AI Agent 中的角色：结构化企业知识的存储层
- 你的实际工作：哪些表？什么业务？数据量级？

**ADF：**
- ADF Pipeline 的核心组件：Linked Service / Dataset / Pipeline / Trigger
- 你设计的 Pipeline：Ingestion → Transformation → Load 的具体逻辑
- 调度策略：日级/小时级触发？出错重试机制？
- ADF 和 Airflow / Dagster 的对比

**Power BI：**
- 你负责的报表：哪个业务场景？哪些指标？
- SOR → Power BI 的 ETL 映射规则具体是什么？
- 数据量和刷新频率？

**SQL：**
- 你写的最复杂的 SQL 是什么？（Window 函数？CTE？）
- 跨境数据的处理：多币种？多时区？

---

## 六、把数据工程经历写成独立项目

> 把江森自控的数据工作单独包装成一个"数据平台"项目，作为 GitHub 仓库。

### 项目名称建议

**企业财务数据分析平台**
或
**Azure 数据工厂金融 ETL 实战**

### 项目描述

> "基于 Azure Data Factory + Snowflake 的企业级财务数据分析平台，实现了从多源业务系统到智能报表的完整数据链路。"

### 技术栈

- Snowflake（企业数据仓库）
- Azure Data Factory（ETL 编排）
- Power BI（数据可视化）
- Python + SQL（数据处理）

### 核心模块（你能写的代码）

1. **ADF Pipeline 配置**：用 YAML 描述 ETL 流程
2. **数据质量监控**：Python 脚本检查数据质量
3. **Snowflake SQL 查询**：复杂财务分析 SQL 示例
4. **数据字典生成器**：从 Snowflake Schema 自动生成 Markdown 文档

### 这个项目的价值

- 展示你的数据工程能力
- 证明你理解企业级数据架构
- 为 AI Agent 提供数据视角的理解
- 和纯 AI 背景的候选人形成差异化

---

## 七、面试高频问题：数据工程版

### Q：数据工程师和 AI 有什么关系？
> "AI 模型的性能上限由数据质量决定。RAG 系统的效果差，80% 是因为数据 pipeline 没做好；Agent 回答不准，往往是底层数据 Schema 不清楚。我在江森自控的工作，做的就是确保 AI 应用的'地基'稳固。"

### Q：你怎么理解 Snowflake 在 AI Agent 中的作用？
> "Snowflake 是企业 AI Agent 的结构化知识库。Agent 的 Tool 实际上就是在查询 Snowflake——Schema 描述是否清晰、数据是否干净、查询性能是否够快，直接决定 Agent 的体验。我在江森自控做的 Schema 优化工作，就是提升 Agent 可用性的关键。"

### Q：ADF 和 RAG 的数据 pipeline 有什么区别？
> "本质一样，形式不同。ADF 处理的是结构化数据（数据库→数据仓库），RAG pipeline 处理的是非结构化数据（文档→向量库）。核心思想一致：数据抽取→清洗→转换→加载。我在 ADF 里积累的 ETL 经验，可以直接迁移到 RAG 数据处理。"

### Q：你觉得 AI 应用落地的瓶颈在哪里？
> "我的经验告诉我，瓶颈在数据层，不在模型层。大多数 AI 应用失败，不是因为模型不够好，而是因为数据质量差、数据更新不及时、数据和业务脱节。我在江森自控深刻理解这一点——Snowflake 仓库的设计、ADF pipeline 的质量控制，都是 AI 应用成功的前置条件。"

---

_本文档帮助王怡琼把江森自控的数据工程经历转化为 AI 方向的竞争力。核心思路：不是"我没有做 AI"，而是"我做了 AI 应用的地基"。_
