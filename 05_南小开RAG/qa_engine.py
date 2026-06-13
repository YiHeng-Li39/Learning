import os
import sys

# 镜像设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['NO_PROXY'] = 'hf-mirror.com'

import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json


class QAEngine:
    def __init__(self, vector_db_path):
        self.vector_db_path = vector_db_path
        self.api_key = None
        self.model = None
        self.collection = None

    def initialize(self):
        if self.model: return
        print("🔧 引擎初始化...")
        try:
            # 保持 BGE-M3 模型不变
            self.model = SentenceTransformer('BAAI/bge-m3', trust_remote_code=True)
            client = chromadb.PersistentClient(path=self.vector_db_path)
            self.collection = client.get_collection("insurance_knowledge")
            print("✅ 引擎就绪 (BGE-M3 已连接)")
        except Exception as e:
            print(f"❌ 引擎初始化错误: {e}")
            raise e

    def set_api_key(self, key):
        self.api_key = key

    def chat_stream(self, query, history=[]):
        if not self.api_key: yield "❌ 请输入 API Key"; return
        if not self.model: self.initialize()

        # --- 1. RAG 检索 ---
        context_str = ""
        try:
            vec = self.model.encode([query]).tolist()
            # 【优化】为了实现更好的“解读”效果，我们将检索片段从 5 提升到 8
            # 这样对于长文献，AI 能看到更多上下文，总结更准确
            res = self.collection.query(query_embeddings=vec, n_results=8)
            docs = res['documents'][0] if res['documents'] else []
            metas = res['metadatas'][0] if res['metadatas'] else []

            context_str = "\n".join([f"📄[{m['filename']}]: {d[:400]}" for d, m in zip(docs, metas)])

            # 推送参考资料折叠卡片
            if docs:
                ref_ui_list = []
                for i, (d, m) in enumerate(zip(docs, metas)):
                    snippet = d[:120].replace('\n', ' ') + "..."
                    ref_ui_list.append(f"**[{i + 1}] {m['filename']}**\n> {snippet}")
                yield f"<references>{chr(10).join(ref_ui_list)}</references>"

        except Exception as e:
            context_str = "(无法连接知识库)"

        # --- 2. System Prompt (新增：文档解读模式) ---
        system_prompt = f"""
你叫“南小开”，是精算与保险领域的智能助手。

【核心能力】
1. **中英双语专家**：你能完美阅读英文文献，并默认用中文回答。
2. **知识问答**：优先基于【参考资料】回答。
3. **复杂计算**：如需计算或制表，请使用 `<run_python>` 包裹 pandas 代码。

【特殊指令：文档解读】
如果用户要求 **“解读”** 或 **“总结”** 某篇特定文档，请按以下结构输出：
1. **📄 核心主题**：一句话概括这篇文献研究了什么。
2. **💡 关键观点/方法**：列出 3-5 个核心论点或研究方法。
3. **📊 结论与意义**：作者得出了什么结论？对行业有什么启示？
(如果参考资料不足以概括全文，请诚实说明“基于现有片段...”)

【格式要求】
- 思考过程请包裹在 <think>...</think> 中。
- 数学公式使用 LaTeX 格式。

【参考资料】
{context_str}
"""

        # --- 3. 构建消息链 ---
        msgs = [{"role": "system", "content": system_prompt}]
        for old_msg in history[-6:]:
            msgs.append(old_msg)
        msgs.append({"role": "user", "content": query})
# --- 4. 请求 DeepSeek ---
        try:
            print(f"⏳ [Debug] 准备发送 DeepSeek API 请求... (使用的Key: {self.api_key[:5]}***)") 
            
            os.environ['HTTP_PROXY'] = ''
            os.environ['HTTPS_PROXY'] = ''
            
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-reasoner", # 👈 核心改变：升级为原生支持深度思考的 R1 模型
                    "messages": msgs,
                    "stream": True
                    # 注意：reasoner 推理模型是由 AI 自动控制思考深度的，官方建议去掉 temperature 等参数
                },
                stream=True,
                timeout=60
            )
            
            print(f"📥 [Debug] 收到响应，状态码: {resp.status_code}")
            
            if resp.status_code != 200:
                error_msg = f"\n\n❌ API 请求失败 (状态码: {resp.status_code})：\n{resp.text}"
                print(f"❌ [Error] {error_msg}")
                yield error_msg
                return

            print("🌊 [Debug] 开始接收流式数据...")
            
            # 用于控制前端 <think> 标签的闭合状态
            has_started_thinking = False
            has_finished_thinking = False

            for line in resp.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    
                    if decoded.startswith(':') or not decoded.strip():
                        continue
                    
                    if decoded.startswith('data: '):
                        json_str = decoded[6:]
                        
                        if json_str.strip() == '[DONE]': 
                            print("\n✅ [Debug] 数据流接收完毕")
                            # 容错处理：如果全是思考没有正文，确保标签闭合
                            if has_started_thinking and not has_finished_thinking:
                                yield "\n</think>\n\n"
                            break
                        
                        try:
                            chunk_json = json.loads(json_str)
                            if 'choices' in chunk_json and len(chunk_json['choices']) > 0:
                                delta = chunk_json['choices'][0].get('delta', {})
                                
                                # 1. 提取并推送原生思考过程 (reasoning_content)
                                reasoning = delta.get('reasoning_content', '')
                                if reasoning:
                                    if not has_started_thinking:
                                        yield "<think>\n" # 动态注入起始标签给前端
                                        has_started_thinking = True
                                    print(reasoning, end="", flush=True)
                                    yield reasoning

                                # 2. 提取并推送正式回答 (content)
                                content = delta.get('content', '')
                                if content:
                                    # 如果思考结束，正文刚开始，注入闭合标签
                                    if has_started_thinking and not has_finished_thinking:
                                        yield "\n</think>\n\n"
                                        has_finished_thinking = True
                                        print("\n[思考结束，开始输出正文]\n", end="", flush=True)
                                        
                                    print(content, end="", flush=True)
                                    yield content
                                    
                        except Exception as parse_err:
                            print(f"\n⚠️ [Warning] JSON解析错误: {parse_err}")
                            
        except Exception as e:
            error_msg = f"\n\n❌ 网络请求或代码执行错误: {e}"
            print(f"💥 [Fatal] {error_msg}")
            yield error_msg