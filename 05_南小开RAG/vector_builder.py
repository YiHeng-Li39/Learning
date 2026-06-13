import os
import sys
import torch  # 引入 torch 用于检测显卡

# ============ 代理与镜像设置 ============
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['NO_PROXY'] = 'hf-mirror.com'

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 只读取预处理后的标准文件
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_pdf")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "high_quality_vector_db")

class VectorDBBuilder:
    def __init__(self):
        print("🔧 初始化入库程序...")
        
        # === 1. 显卡检测逻辑 ===
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("==================================================")
        if self.device == "cuda":
            print(f"🚀 检测到 NVIDIA 显卡: {torch.cuda.get_device_name(0)}")
            print("✅ 已启用 GPU 加速模式 (速度将提升 10-50 倍)")
        else:
            print("⚠️  未检测到可用显卡，将使用 CPU 慢速运行")
            print("💡 提示: 如果你有 N 卡但显示此提示，请检查 PyTorch 是否安装了 GPU 版")
        print("==================================================")

        try:
            print("📥 正在检查/加载 BGE-M3 模型...")
            model_dir = snapshot_download(
                repo_id="BAAI/bge-m3",
                resume_download=True,
                max_workers=4,
                ignore_patterns=["*.DS_Store", "imgs/*"] 
            )
            
            # === 2. 加载模型时指定设备 ===
            self.model = SentenceTransformer(model_dir, trust_remote_code=True, device=self.device)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)

        self.client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(name="insurance_knowledge")
        
        try:
            # 只读取 ids 用于比对，节省内存
            existing = self.collection.get(include=['ids'])
            self.existing_ids = set(existing['ids']) if existing['ids'] else set()
            print(f"📊 当前库存: {len(self.existing_ids)} 条")
        except Exception:
            self.existing_ids = set()

    def run(self):
        docs, metas, ids = [], [], []
        
        # 1. 扫描 processed_pdf
        file_list = []
        if not os.path.exists(PROCESSED_DIR):
            print(f"❌ 路径不存在: {PROCESSED_DIR}")
            return

        for root, dirs, files in os.walk(PROCESSED_DIR):
            for file in files:
                if file.endswith('.txt'):
                    file_list.append((root, file))

        if not file_list:
            print("❌ processed_pdf 为空！请先运行第一段代码。")
            return

        print(f"📂 发现 {len(file_list)} 个预处理片段，正在快速对比增量...")
        
        # 2. 读取文件 (跳过已存在的)
        for root, file in tqdm(file_list, desc="比对进度"):
            unique_id = f"{os.path.basename(root)}_{file}"
            if unique_id in self.existing_ids:
                continue

            try:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read().split('\n\n', 1)
                    if len(content) > 1:
                        docs.append(content[1])
                        metas.append({"filename": file, "category": os.path.basename(root)})
                        ids.append(unique_id)
            except Exception:
                pass

        if not docs:
            print("✅ 知识库已是最新，无需更新。")
            return

        # 3. 写入数据库
        print(f"🚀 正在为 {len(docs)} 条新数据生成向量并写入...")
        
        # 根据设备调整批处理大小，GPU 可以一次处理更多
        batch_size = 32 if self.device == "cuda" else 8
        
        for i in tqdm(range(0, len(docs), batch_size), desc="入库进度 (GPU)" if self.device == "cuda" else "入库进度 (CPU)"):
            end = i + batch_size
            try:
                self.collection.add(
                    embeddings=self.model.encode(docs[i:end]).tolist(),
                    documents=docs[i:end],
                    metadatas=metas[i:end],
                    ids=ids[i:end]
                )
            except Exception as e:
                print(f"⚠️ 批量写入失败: {e}")

        print(f"🎉 全部完成！当前总计: {self.collection.count()} 条")

if __name__ == "__main__":
    VectorDBBuilder().run()