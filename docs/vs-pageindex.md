# pidx 与 PageIndex 对比

整理于 2026-11-03。PageIndex 一方的信息来自它公开的网站、文档、更新日志和开源仓库（链接见文末），没有用过它的 Cloud 账号。
pidx 一方是当前 `main` 分支的实际行为。

> **还没有全面评测。** 下面的性能数字，pidx 一方是在我们自己的小样本上测的（几十个问题、两个资料包），
> PageIndex 一方是它自己公布的数字，用的是它自己的数据集。两边的数据和指标不同，只能看量级，不能直接比高下。
> 要下结论，需要在同一个数据集上测证据页召回和答案正确率，见文末「下一步」。

## PageIndex 现在有哪些产品

| 产品 | 是什么 |
|---|---|
| **开源 / Local**（`pip install pageindex`） | 在本机建树索引（默认用 Flash，不靠 LLM 的版面规则），只支持文本型 PDF，页级引用 |
| **Cloud** | 托管的建索引和存储：支持扫描件和图片多的文档（OCR + 图像理解），块级引用（带坐标框），元数据，文件夹，远程 MCP；按页计费 |
| **Enterprise** | Cloud 部署到客户自己的 VPC（AWS / Azure / GCP） |
| **Page Search API** | 给问题返回带分数的页面排名，约 1 秒；一次最多 10 个问题；能搜照片、图表；可按元数据过滤 |
| **Chat App** | 面向终端用户的长文档问答，引用精确到行 |
| **PageIndex File System**（Cloud） | 文件级的树索引层，让推理覆盖整个文档库，而不只是单个文档 |

## 能力对比

图例：✅ 有 · 🟡 部分有 · ❌ 没有

### 建索引

| 能力 | PageIndex | pidx | 说明 |
|---|---|---|---|
| 文本型 PDF 建树 | ✅ Flash（规则，快）/ classic（LLM） | ✅ | pidx 依次尝试：书签 → 版面候选交 LLM 定结构 → LLM 读页；超长节点递归拆分 |
| 扫描件 / 图片 OCR | ✅ Cloud | ✅ | pidx 用 mineru.net 云端 API |
| Markdown | ✅ 开源版有 `page_index_md` | ✅ | pidx 2026-11-03 加入，本地解析 |
| Word / PPT / Excel | 未见文档提及 | ✅ | 交给 MinerU 原生解析，切成虚拟页（没有版面坐标） |
| zip / 目录结构 | 🟡 文件夹（用 API 建），File System | ✅ | pidx 上传 zip，目录成为树的上层节点，每个目录有 LLM 写的摘要 |
| **图像理解** | ✅ Cloud：保存图片，agent 取图交给视觉模型 | ❌ | pidx 只有 OCR 出来的文字和图注。截图里的面板、图表里的数字经常丢 |
| 建索引过程可见 | 未见 | ✅ | 处理时间线：每个文件的排队、解析、建树、摘要、向量各花多久 |

### 检索与问答

| 能力 | PageIndex | pidx | 说明 |
|---|---|---|---|
| LLM 在树上推理检索 | ✅ agent 调工具逐步读 | ✅ | pidx 默认先用节点向量召回几棵子树，LLM 并行读它们；也可以读整棵树（`retrieval=tree`）；证据不足会自动补搜一轮 |
| **页面检索（Page Search）** | ✅ 约 1 秒，0–1 相关分，可搜图 | 🟡 | pidx 2026-11-03 加了 `POST /v1/pages/search`：**只用向量打分**，不调 LLM。还没有 LLM 精排，不能搜图 |
| 多文档 / 全库 | ✅ | ✅ | pidx 先按向量给文档排序，再由 LLM 读文档描述选文档 |
| **元数据过滤** | ✅ 文件夹声明带类型的字段，支持 `$gte` 这类比较 | 🟡 | pidx 上传时能带 metadata；只有页面检索支持精确匹配过滤，没有字段类型和比较运算 |
| 块级引用 + 原页高亮 | ✅ Cloud（2026-09 起） | ✅ | PDF 和图片有坐标框高亮；Office / Markdown 只能标出文本 |
| **多轮对话** | ✅ `chat()` / Chat App | ❌ | pidx 每次检索都是单轮 |
| 流式输出 | ✅ | ✅ | pidx 用 SSE 推送每一步（选文档、读子树、作答）和耗时瀑布图 |
| 换模型不重建索引 | ✅ | ✅ | pidx 可以按请求选检索模型 |

### 接口与集成

| 能力 | PageIndex | pidx | 说明 |
|---|---|---|---|
| REST API | ✅ | ✅ | 接口形状参照了 PageIndex（submit / tree / page content / block） |
| Python SDK | ✅ | ❌ | pidx 只有 REST 和 CLI |
| JS SDK | ✅ | ❌ | |
| MCP | ✅ 远程 HTTP，Bearer key | 🟡 | pidx 只有本地 stdio MCP（7 个工具，含 `search_pages`）；没有远程 MCP，也没有按用户的 key |
| Agent 框架集成 | ✅ OpenAI Agents、Claude Agent SDK、Vercel AI、LangChain | 🟡 | 只能通过 MCP 接入 |
| 取图工具（`get_document_image`） | ✅ | ❌ | 和图像理解是同一个缺口 |

### 部署与商业化

| 能力 | PageIndex | pidx |
|---|---|---|
| 自托管 | ✅ 开源版（功能有限）/ 企业 VPC | ✅ 全部功能都能自托管 |
| 多租户、API key、计费 | ✅ | ❌ 只有一个访问 token |
| 公开评测 | ✅ 金融问答 benchmark（40 份 10-K、100 题、19 个模型）、page-search 召回评测 | ❌ |

## 性能对比（不是同一数据集，只看量级）

| 指标 | PageIndex（官方公布） | pidx（我们自测） |
|---|---|---|
| 页面检索延迟 | 0.89 秒（93 页的财报）；新的 160 页文档约 1.1 秒，含处理 | 服务端中位数 **171 ms**、p90 188 ms，全库 32 份文档 629 页（2026-11-03，线上实测）；一次 8 个问题 235 ms。**前提是文档已经建好索引** |
| 新文档多久能搜 | Page Search：约 1 秒，含处理 | 要等完整建索引：37 页论文 41 秒，50 页 47 秒，27 页 1 分 13 秒；15 个文件的资料包 5 分 45 秒（2 个并发） |
| 页面召回（证据页落在前 k 名的比例） | top5 93.2%，top10 97.5%，top20 99.3%（同一测试里三家向量检索 top5 为 47.5–65.5%，top10 为 68.0–83.8%） | **未测**。现在的 pidx 页面检索就是纯向量，按他们的数据，纯向量检索比他们的 Page Search 低一大截，可以预期我们也不会高 |
| 问答正确率 | 金融问答 benchmark，按模型公布准确率和成本 | 只有小样本：`scripts/eval_formats.py` 的 14 题全部通过，13 个演示问题人工核对过；PRH 资料包 8 题有 7 题答出关键信息 |
| 问答延迟 / token | 未公布 | DeepSeek V4 Flash，13 个演示问题平均 3.6 秒、约 1.6 万 token（混合检索）；纯树检索 4.0 秒、约 3 万 token |
| 费用 | 建索引 $0.01/页；保持可检索 $0.001/页/月；Page Search 每 1000 页首个问题 $0.10 | 自托管：建索引用免费的 Atria，向量用 SiliconFlow，OCR 用 mineru.net 的免费额度；问答按 DeepSeek token 计费 |

## 主要差距（按优先级）

1. **评测**：没有在统一数据集上测过证据页召回和答案正确率，其他各项改进是否有效都说不清。
2. **页面检索的质量和首次可用时间**：目前只用向量打分；没有 LLM 精排；新文档必须等完整建树，不能像他们那样约 1 秒就能搜。
3. **图像理解**：没有视觉模型，图表、截图里的信息只能靠 OCR。需要付费的视觉模型（例如 DeepInfra 上的 Qwen3-VL）。
4. **元数据和文件夹**：缺少带类型的元数据字段和比较过滤；不能用 API 建资料包、往资料包里加文件。
5. **接口形态**：远程 MCP + 鉴权、多轮对话、Python SDK。工作量都不大。
6. 多租户、计费：自用场景暂不需要。

## pidx 做得更多的地方

- 输入格式：Office、Markdown、zip 目录结构，目录有摘要。
- 检索过程全程可见：每一步的推理说明、召回了哪几棵子树、LLM 调用的耗时瀑布图。
- 建索引过程可见：资料包处理时间线。
- 混合检索（向量召回子树 + LLM 读子树）和纯 LLM 读树可以按请求切换，方便做 A/B。

## 下一步：统一评测

- 从星河、PRH 两个资料包各标 30–50 个问题，每题标出证据页。另外可以借用 PageIndex 公开的金融问答集（40 份 10-K、100 题）。
- 指标：页面召回 @5 / @10 / @20、答案正确率、延迟、token 数。
- 对照组：纯向量页面检索、混合检索、纯树检索，以及开源 PageIndex。

## 来源

- PageIndex 主页 <https://pageindex.ai>，Developer 与定价 <https://pageindex.ai/developer>，Benchmark <https://pageindex.ai/benchmark>
- Page Search API <https://search.pageindex.ai>
- 文档 <https://docs.pageindex.ai>：Getting Started、MCP、Changelog、Cookbook（Multimodal、Metadata）
- 开源仓库 <https://github.com/VectifyAI/PageIndex>（README 的 Local / Cloud 对照表、Financial QA Benchmark）
