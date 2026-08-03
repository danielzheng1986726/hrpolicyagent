import os
import re
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from rag import search, build_index

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时检查并构建索引
    index_path = "data/faiss_index/index.faiss"
    if not os.path.exists(index_path):
        logger.info("索引不存在，正在构建...")
        try:
            build_index()
            logger.info("索引构建完成")
        except Exception as e:
            logger.error(f"索引构建失败: {str(e)}")
            # 不阻止启动，但记录错误
    else:
        logger.info("索引已存在，跳过构建")
    yield

# 初始化模板
templates = Jinja2Templates(directory="templates")

# 初始化 OpenAI 客户端（兼容本地和云端环境变量）
api_key = os.getenv("AI_BUILDER_TOKEN") or os.getenv("AIB_API_KEY")
if not api_key:
    raise ValueError("AI_BUILDER_TOKEN 或 AIB_API_KEY 环境变量未设置，请在 .env 文件中配置或设置环境变量")

client = OpenAI(
    api_key=api_key,
    base_url="https://space.ai-builders.com/backend/v1"
)

app = FastAPI(title="HR政策助手 API", lifespan=lifespan)

# 系统提示词
SYSTEM_PROMPT = """你是星辰科技有限公司的HR政策顾问，帮助员工解答公司政策问题。

你有一个工具可以使用：
- 查询政策库：当员工问到具体政策细节时（如年假天数、病假工资、离职流程等），必须使用格式 [SEARCH: 关键词] 来查询

重要规则：
1. 如果是闲聊或通用问题（如"你好"、"谢谢"、"再见"），直接回复，不要使用 [SEARCH: xxx]
2. 如果涉及具体政策细节（年假、病假、工资、离职、试用期等），必须先输出 [SEARCH: 关键词]，然后等待查询结果
3. 查询格式必须严格遵循：[SEARCH: 关键词]，例如 [SEARCH: 病假工资] 或 [SEARCH: 年假天数]
4. 不要在没有查询的情况下直接回答政策细节问题
5. 查询关键词要简洁精准，如"病假工资"、"年假天数"、"离职通知期"
6. 用户可能基于上文追问（如"那病假呢？""不满一年的呢？"），要结合对话历史补全用户真正想问的内容，再决定查询关键词

工作流程示例：
用户问："病假工资怎么算？"
你应该回复："[SEARCH: 病假工资]"
然后等待查询结果，再基于结果回答。

用户问："你好"
你应该直接回复问候，不要使用 [SEARCH: xxx]"""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []

class ChatResponse(BaseModel):
    reply: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页，返回聊天界面"""
    return templates.TemplateResponse(request, "index.html")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天端点，实现ReAct模式：AI自主决定是否需要查询政策库"""
    try:
        logger.info(f"收到用户问题: {request.message}")

        # 多轮记忆:接收前端带来的最近对话历史(服务器无状态,记忆保存在用户浏览器)
        # 只接受 user/assistant 角色,最多保留最近 6 条,单条截断到 2000 字符
        history = [
            {"role": m.role, "content": m.content[:2000]}
            for m in request.history
            if m.role in ("user", "assistant")
        ][-6:]
        if history:
            logger.info(f"携带对话历史 {len(history)} 条")

        # 第一次调用AI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": request.message}
        ]
        
        response = client.chat.completions.create(
            model="deepseek",
            messages=messages,
            max_tokens=1000
        )
        
        first_reply = response.choices[0].message.content
        logger.info(f"AI首次回复: {first_reply[:100]}...")
        
        # 检查是否包含 [SEARCH: xxx] 格式
        search_pattern = r'\[SEARCH:\s*([^\]]+)\]'
        search_match = re.search(search_pattern, first_reply)
        
        if not search_match:
            # 不需要查询，直接返回
            logger.info("未触发检索，直接返回AI回复")
            return {"reply": first_reply}
        
        # 提取查询关键词
        search_keyword = search_match.group(1).strip()
        logger.info(f"触发检索，关键词: {search_keyword}")
        
        # 执行RAG检索
        try:
            search_results = search(search_keyword, top_k=3)
            logger.info(f"检索到 {len(search_results)} 个相关chunks")
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return {"reply": "抱歉，查询政策库时出现错误，请稍后重试。"}
        
        # 构造检索结果文本
        search_content = ""
        for i, result in enumerate(search_results, 1):
            search_content += f"---\n{result['content']}\n---\n"
        
        # 第二次调用AI，基于检索结果生成最终回答
        messages.append({"role": "assistant", "content": first_reply})
        messages.append({
            "role": "user",
            "content": f"""查询结果：
{search_content}
请基于以上政策内容回答用户问题。如果查询结果中没有相关信息，请如实告知。"""
        })
        
        logger.info("基于检索结果生成最终回答...")
        final_response = client.chat.completions.create(
            model="deepseek",
            messages=messages,
            max_tokens=1500
        )
        
        final_reply = final_response.choices[0].message.content
        logger.info("生成最终回答完成")
        
        return {"reply": final_reply}
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"处理请求时出错: {error_msg}")
        # 提供更友好的错误提示
        if "Connection" in error_msg or "connect" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail="无法连接到AI服务，请检查网络连接或API配置"
            )
        elif "401" in error_msg or "Unauthorized" in error_msg:
            raise HTTPException(
                status_code=401,
                detail="API Key 无效，请检查环境变量 AI_BUILDER_TOKEN 或 AIB_API_KEY 配置"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"AI API 调用失败: {error_msg}"
            )
