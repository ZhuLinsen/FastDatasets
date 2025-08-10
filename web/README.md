# FastDatasets Web Interface

基于 Gradio 的智能数据集生成工具 Web 界面

## 🌟 功能特性

- 📤 **文件上传**: 支持 PDF, DOCX, TXT, Markdown 格式
- ⚡ **异步处理**: 支持多文件并发处理，实时进度监控
- 🎛️ **参数配置**: 灵活的文本分块、LLM 配置等参数
- 📊 **进度监控**: 实时查看处理进度和状态
- 📁 **结果管理**: 查看和下载生成的数据集文件
- 🤖 **多格式导出**: 支持 Alpaca、ShareGPT 等格式
- 🧠 **思维链支持**: 可选启用 Chain of Thought (CoT)

## 🚀 快速开始

### 1. 安装依赖

```bash
cd web
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
# LLM API 配置
API_KEY=your_api_key_here
BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-3.5-turbo

# 其他配置
CHUNK_MIN_LEN=200
CHUNK_MAX_LEN=1000
OUTPUT_FORMATS=alpaca,sharegpt
ENABLE_COT=true
MAX_LLM_CONCURRENCY=3
MAX_FILE_CONCURRENCY=2
```

### 3. 启动服务

#### 方式一：使用启动脚本（推荐）

```bash
python run.py
```

#### 方式二：直接运行

```bash
python app.py
```

### 4. 访问界面

打开浏览器访问：http://localhost:7860

## 📖 使用指南

### 文档处理流程

1. **上传文件**
   - 在「📁 文档处理」标签页选择要处理的文件
   - 支持多文件同时上传
   - 支持的格式：PDF, DOCX, TXT, Markdown

2. **配置参数**
   - **文本分块参数**：设置最小/最大块长度
   - **输出格式**：选择 Alpaca 和/或 ShareGPT 格式
   - **并发设置**：调整 LLM 和文件处理并发数
   - **LLM 配置**：设置 API Key、Base URL、模型名称

3. **开始处理**
   - 点击「🚀 开始处理」按钮
   - 系统将异步处理所有文件

4. **监控进度**
   - 切换到「📊 进度监控」标签页
   - 实时查看处理进度和状态
   - 自动刷新，无需手动操作

5. **查看结果**
   - 在「📁 结果查看」标签页查看处理结果
   - 显示生成的问答对数量和文件路径
   - 可以下载生成的数据集文件

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 最小块长度 | 文本分块的最小字符数 | 200 |
| 最大块长度 | 文本分块的最大字符数 | 1000 |
| 输出格式 | 数据集导出格式 | alpaca, sharegpt |
| 启用 CoT | 是否启用思维链推理 | true |
| LLM 并发数 | 同时调用 LLM 的最大数量 | 3 |
| 文件并发数 | 同时处理文件的最大数量 | 2 |

## 🔧 高级配置

### 自定义端口

修改 `app.py` 或 `run.py` 中的端口设置：

```python
interface.launch(
    server_port=8080,  # 修改为你想要的端口
    # ... 其他参数
)
```

### 外网访问

如需外网访问，设置 `share=True`：

```python
interface.launch(
    share=True,  # 启用公网访问
    # ... 其他参数
)
```

### Docker 部署

创建 `Dockerfile`：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 复制项目文件
COPY . .

# 安装依赖
RUN pip install -r web/requirements.txt

# 暴露端口
EXPOSE 7860

# 启动服务
CMD ["python", "web/run.py"]
```

构建和运行：

```bash
docker build -t fastdatasets-web .
docker run -p 7860:7860 -v $(pwd)/output:/app/output fastdatasets-web
```

## 🐛 故障排除

### 常见问题

1. **依赖安装失败**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **端口被占用**
   ```bash
   # 查看端口占用
   lsof -i :7860
   # 或修改为其他端口
   ```

3. **LLM API 调用失败**
   - 检查 API Key 是否正确
   - 确认 Base URL 是否可访问
   - 验证模型名称是否正确

4. **文件处理失败**
   - 确认文件格式是否支持
   - 检查文件是否损坏
   - 查看日志文件获取详细错误信息

### 日志查看

日志文件位置：`../logs/`

```bash
# 查看最新日志
tail -f ../logs/app.log
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。