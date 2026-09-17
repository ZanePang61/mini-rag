# 小型轻量化RAG检索增强生成系统

## 项目简介

一款小而精的轻量化RAG（Retrieval-Augmented Generation）系统，支持 PDF、Word、Excel 三类办公文档的解析入库、语义检索、智能问答全流程功能。

## 技术选型

| 组件 | 技术方案 | 说明 |
|------|---------|------|
| 文档解析 | PyMuPDF / python-docx / openpyxl | 分别处理PDF、Word、Excel |
| 文本分块 | 自研段落感知分块算法 | 支持固定长度+重叠补全 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | 开源中文嵌入模型，CPU友好 |
| 向量数据库 | ChromaDB | 本地持久化，支持元数据过滤 |
| 大模型 | DeepSeek Chat API | 通过OpenAI兼容接口调用 |
| Web界面 | Streamlit | 轻量Python Web框架 |

## 环境要求

- Python 3.9+
- CPU即可运行（无需GPU）
- 内存 4GB+

## 安装步骤

```bash
# 1. 进入项目目录
cd E:\mini-rag

# 2. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
pip install python-dotenv
```

## 运行方式

```bash
# 启动Web界面
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

## 使用流程

1. **上传文档**：在「文档上传与解析」页面选择 PDF/Word/Excel 文件
2. **自动处理**：系统自动完成解析→清洗→分块→向量化→入库
3. **智能问答**：在「智能问答」页面输入问题，系统检索相关片段并生成回答
4. **结果溯源**：回答附带来源文件名和检索片段预览

## 项目结构

```
mini-rag/
├── app.py                  # Streamlit Web入口
├── .env                    # 环境变量配置
├── requirements.txt        # Python依赖
├── README.md               # 说明文档
├── src/
│   ├── __init__.py
│   ├── parser.py           # 文档解析（PDF/Word/Excel）
│   ├── chunker.py          # 文本清洗+智能分块
│   ├── embedder.py         # 向量嵌入（bge-small-zh）
│   ├── retriever.py        # 向量存储与检索（ChromaDB）
│   └── rag.py              # RAG问答组装（DeepSeek API）
├── chroma_db/              # ChromaDB持久化数据
└── data/                   # 上传文件临时存储
```

## 核心设计思路

### 文档解析
- **PDF**：使用PyMuPDF逐页提取文本，保留段落逻辑，去除乱码和空白
- **Word**：使用python-docx提取正文段落和表格内容，过滤格式标记
- **Excel**：使用openpyxl按工作表逐行读取，保留单元格对应关系

### 文本分块
采用**段落感知分块**策略：优先按段落（双换行符）切分，段落过长时再按固定长度切分，相邻块之间有50字符重叠，避免上下文断裂。

### 向量处理
- 嵌入模型：bge-small-zh-v1.5（512维），中文语义理解优秀，模型仅约100MB
- 向量库：ChromaDB，cosine相似度，本地持久化
- 文档区分：每个块附带filename元数据，检索结果可溯源

### RAG问答
- 系统提示词严格约束模型只基于文档内容回答
- 无相关信息时主动说明，杜绝幻觉
- 回答标注信息来源

## 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| chunk_size | 500 | 分块字符数 |
| chunk_overlap | 50 | 分块重叠字符数 |
| top_k | 5 | 检索召回数量 |
| temperature | 0.3 | 大模型回答温度 |

## 存在问题与优化方向

1. **PDF扫描件**：当前仅支持文本型PDF，扫描件需集成OCR（如PaddleOCR）
2. **大文件处理**：未实现异步处理，大文件上传可能阻塞界面
3. **多轮对话**：当前为单轮问答，可扩展为多轮上下文对话
4. **Excel结构化查询**：可增加SQL-like查询能力，针对表格数据做统计分析
5. **权限管理**：无用户认证，适合本地单用户使用
