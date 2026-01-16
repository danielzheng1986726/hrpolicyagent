#!/usr/bin/env python3
"""
索引构建脚本
运行此脚本将读取员工手册，生成embeddings并建立FAISS索引
"""

from rag import build_index

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 开始构建RAG索引")
    print("=" * 60)
    
    try:
        index, chunks = build_index()
        print("\n" + "=" * 60)
        print("✅ 索引构建完成！")
        print(f"   - 共处理 {len(chunks)} 个文档chunks")
        print(f"   - 索引包含 {index.ntotal} 个向量")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 索引构建失败: {str(e)}")
        print("=" * 60)
        raise
