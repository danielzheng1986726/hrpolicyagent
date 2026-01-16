import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置路径
DATA_DIR = Path(__file__).parent / "data"
HANDBOOK_PATH = DATA_DIR / "hr_policy_handbook.md"
CHUNKS_JSON_PATH = DATA_DIR / "chunks.json"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# 初始化 OpenAI 客户端（兼容本地和云端环境变量）
api_key = os.getenv("AI_BUILDER_TOKEN") or os.getenv("AIB_API_KEY")
if not api_key:
    raise ValueError("AI_BUILDER_TOKEN 或 AIB_API_KEY 环境变量未设置，请在 .env 文件中配置或设置环境变量")

client = OpenAI(
    api_key=api_key,
    base_url="https://space.ai-builders.com/backend/v1"
)


def load_and_split_documents() -> List[Dict[str, str]]:
    """
    读取员工手册，按二级标题（## ）切分成chunks
    每个二级标题及其下的所有内容（包括所有三级标题）作为一个chunk
    
    Returns:
        List[Dict]: 包含 id, title, content 的chunks列表
    """
    if not HANDBOOK_PATH.exists():
        raise FileNotFoundError(f"员工手册文件不存在: {HANDBOOK_PATH}")
    
    with open(HANDBOOK_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = []
    # 使用正则表达式匹配所有二级标题（## 开头，后面跟非#字符）
    # 模式：## 后面跟标题，然后是内容（直到下一个##或文档结束）
    pattern = r'##\s+(.+?)(?=\n##\s+|$)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for idx, match in enumerate(matches):
        full_section = match.group(0)
        # 提取标题（第一行，去掉##）
        title_line = match.group(1).split('\n')[0].strip()
        title = title_line
        
        # 提取内容（去掉标题行和分隔线）
        content_lines = full_section.split('\n')[1:]  # 跳过标题行
        content_text = '\n'.join(content_lines).strip()
        
        # 移除分隔线（---）
        content_text = '\n'.join([line for line in content_text.split('\n') 
                                  if not line.strip().startswith('---') and line.strip()])
        
        # 只保留有实际内容的chunks（内容长度大于10字符）
        if len(content_text) > 10:
            chunks.append({
                "id": f"chunk_{idx}",
                "title": title,
                "content": content_text
            })
    
    return chunks


def get_embedding(text: str, max_retries: int = 3) -> List[float]:
    """
    调用 AI Builder Space 的 embedding API 获取文本向量
    
    Args:
        text: 要生成embedding的文本
        max_retries: 最大重试次数
    
    Returns:
        List[float]: embedding向量
    """
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 指数退避：2秒、4秒、6秒
                print(f"Embedding API 调用失败，{wait_time}秒后重试... (尝试 {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise Exception(f"Embedding API 调用失败，已重试 {max_retries} 次: {str(e)}")
    
    raise Exception("无法获取embedding")


def build_index():
    """
    为所有chunks生成embedding，建立FAISS索引并保存
    """
    print("📖 正在读取并切分文档...")
    chunks = load_and_split_documents()
    print(f"✅ 文档切分完成，共 {len(chunks)} 个chunks")
    
    # 保存chunks到JSON
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ Chunks已保存到 {CHUNKS_JSON_PATH}")
    
    # 生成embeddings
    print("\n🔄 正在生成embeddings...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"  正在处理第 {i+1}/{len(chunks)} 个chunk: {chunk['title']}")
        embedding = get_embedding(chunk['content'])
        embeddings.append(embedding)
        # 避免API限流，添加短暂延迟
        if i < len(chunks) - 1:
            time.sleep(0.5)
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"✅ Embeddings生成完成，维度: {embeddings_array.shape}")
    
    # 建立FAISS索引
    print("\n🔍 正在建立FAISS索引...")
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)  # 使用L2距离
    index.add(embeddings_array)
    print(f"✅ FAISS索引建立完成，包含 {index.ntotal} 个向量")
    
    # 保存索引
    index_path = FAISS_INDEX_DIR / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"✅ 索引已保存到 {index_path}")
    
    # 保存embedding维度信息
    dimension_path = FAISS_INDEX_DIR / "dimension.txt"
    with open(dimension_path, 'w') as f:
        f.write(str(dimension))
    
    return index, chunks


def load_index() -> Tuple[faiss.Index, List[Dict[str, str]]]:
    """
    加载已保存的FAISS索引和chunks
    
    Returns:
        Tuple[faiss.Index, List[Dict]]: FAISS索引和chunks列表
    """
    index_path = FAISS_INDEX_DIR / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"索引文件不存在: {index_path}，请先运行 build_index.py")
    
    if not CHUNKS_JSON_PATH.exists():
        raise FileNotFoundError(f"Chunks文件不存在: {CHUNKS_JSON_PATH}，请先运行 build_index.py")
    
    # 加载索引
    index = faiss.read_index(str(index_path))
    
    # 加载chunks
    with open(CHUNKS_JSON_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    return index, chunks


def search(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    检索与查询最相关的chunks
    
    Args:
        query: 用户查询问题
        top_k: 返回最相关的k个结果
    
    Returns:
        List[Dict]: 包含 id, title, content 的相关chunks列表
    """
    # 加载索引和chunks
    index, chunks = load_index()
    
    # 获取查询的embedding
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding], dtype=np.float32)
    
    # 搜索最相似的chunks
    distances, indices = index.search(query_vector, top_k)
    
    # 返回结果
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(chunks):
            result = chunks[idx].copy()
            result['distance'] = float(distance)  # 添加距离信息
            results.append(result)
    
    return results
