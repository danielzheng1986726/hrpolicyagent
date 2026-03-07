# HR Policy Assistant

基于 RAG 的 HR 政策问答助手，通过检索增强生成技术回答公司政策问题。

## 技术栈
- FastAPI + Jinja2 (Web 服务与模板渲染)
- FAISS (向量检索)
- OpenAI API (大语言模型)
- Python 3

## 使用方法
```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env  # 填入 AI_BUILDER_TOKEN 或 AIB_API_KEY

# 启动服务
uvicorn app:app --reload
```

访问 http://localhost:8000 进入聊天界面，输入 HR 政策相关问题即可获得回答。
