FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动应用（使用 PORT 环境变量）
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"
