import os
import fitz  # PyMuPDF
from datetime import datetime
import time
import sys
import numpy as np
import gc
from PIL import Image
import warnings

# ============ 0. Python 3.9 兼容性补丁 ============
if sys.version_info < (3, 10):
    try:
        import importlib_metadata
        import importlib.metadata
        if not hasattr(importlib.metadata, 'packages_distributions'):
            importlib.metadata.packages_distributions = importlib_metadata.packages_distributions
    except ImportError:
        pass

warnings.filterwarnings("ignore")

# ============ 依赖检测 ============
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ 警告: 未检测到 pandas，无法处理 Excel 文件")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️ 警告: 未检测到 pdf2image，OCR功能受限")

try:
    import easyocr
    EASY_OCR_AVAILABLE = True
except ImportError:
    EASY_OCR_AVAILABLE = False
    print("⚠️ 警告: 未检测到 easyocr")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class UniversalFileProcessor:
    def __init__(self):
        # 路径保持原样
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.raw_base_dir = os.path.join(self.base_dir, "data", "raw_pdf")
        self.processed_base_dir = os.path.join(self.base_dir, "data", "processed_pdf")

        os.makedirs(self.processed_base_dir, exist_ok=True)
        os.makedirs(self.raw_base_dir, exist_ok=True)

        self.reader = None
        # 【保留】OCR 初始化逻辑
        if EASY_OCR_AVAILABLE:
            try:
                print("🔄 正在初始化 EasyOCR (尝试使用 GPU 加速)...")
                self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
                print("✅ EasyOCR 初始化成功 (GPU)")
            except Exception as e:
                print(f"⚠️ GPU 初始化失败，切换回 CPU 模式: {e}")
                self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

        self.stats = {"total": 0, "skipped": 0, "success": 0, "failed": 0}

    # ============ 功能 1：处理 Excel (逻辑完全保留) ============
    def process_excel(self, file_path):
        if not PANDAS_AVAILABLE: return ""
        full_text = ""
        try:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
                df = df.fillna(" ")
                if len(df) > 300:
                    markdown_table = df.head(300).to_markdown(index=False)
                    markdown_table += f"\n\n(注：表格过长，仅截取前300行，共 {len(df)} 行)"
                else:
                    markdown_table = df.to_markdown(index=False)
                full_text += f"\n\n=== Excel工作表: {sheet_name} ===\n{markdown_table}\n"
            return full_text
        except Exception as e:
            print(f"      ❌ Excel 解析失败: {e}")
            return ""

    # ============ 功能 2：PDF 智能路由 (保留) ============
    def analyze_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text_len = sum([len(page.get_text()) for page in doc[:3]])
            doc.close()
            return "text" if text_len > 50 else "scan"
        except:
            return "unknown"

    # ============ 功能 3：EasyOCR 核心 (保留) ============
    def run_easyocr(self, image):
        if not self.reader: return ""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            width, height = image.size
            if width > 2000 or height > 2000:
                ratio = min(2000 / width, 2000 / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img_array = np.array(image)
            result = self.reader.readtext(img_array, detail=0, paragraph=True)
            return "\n".join(result)
        except Exception as e:
            return ""

    # ============ 功能 4：OCR 流程控制 (保留) ============
    def process_ocr(self, pdf_path):
        if not PDF2IMAGE_AVAILABLE: return ""
        full_text = ""
        try:
            doc = fitz.open(pdf_path)
            total = len(doc)
            doc.close()
            print(f"      >> 🖼️ 启动 EasyOCR 识别 (共 {total} 页)...")
            pbar = tqdm(total=total, unit="页", ncols=80) if TQDM_AVAILABLE else None
            batch_size = 5
            for start in range(1, total + 1, batch_size):
                end = min(start + batch_size - 1, total)
                images = convert_from_path(pdf_path, first_page=start, last_page=end, dpi=200, thread_count=4)
                for img in images:
                    text = self.run_easyocr(img)
                    if text: full_text += text + "\n"
                    if pbar: pbar.update(1)
                del images
                gc.collect()
            if pbar: pbar.close()
        except Exception as e:
            print(f"      ❌ OCR 中断: {e}")
        return full_text

    # ============ 功能 5：PDF 纯文本提取 (保留) ============
    def process_text(self, pdf_path):
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc: text += page.get_text() + "\n"
            doc.close()
        except:
            pass
        return text

    # ============ 功能 6：文本切片 (保留) ============
    def chunk_text(self, text, chunk_size=800, overlap=100):
        if not text: return []
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                newline_pos = text[start:end].rfind('\n')
                if newline_pos > chunk_size * 0.7: end = start + newline_pos + 1
            chunk = text[start:end].strip()
            if len(chunk) > 20: chunks.append(chunk)
            step = chunk_size - overlap
            if step <= 0: step = chunk_size
            start += step
        return chunks

    def run(self):
        print("==========================================")
        print("🚀 南小开全能知识库处理程序 V7.0 (含TXT)")
        print("==========================================\n")

        if not os.path.exists(self.raw_base_dir):
            print(f"❌ 错误: 原始文件夹不存在: {self.raw_base_dir}")
            print(f"👉 请手动创建并放入文件")
            return

        for category in os.listdir(self.raw_base_dir):
            cat_path = os.path.join(self.raw_base_dir, category)
            if not os.path.isdir(cat_path): continue

            out_dir = os.path.join(self.processed_base_dir, category)
            os.makedirs(out_dir, exist_ok=True)

            print(f"📂 扫描分类: 【{category}】")
            # 【修改 1】这里加入了 .txt
            files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.pdf', '.xlsx', '.xls', '.txt'))]

            for file in files:
                self.stats['total'] += 1
                base_name = os.path.splitext(file)[0]

                # 【增量检查】保留原逻辑
                check_path = os.path.join(out_dir, f"{base_name}_part0.txt")
                if os.path.exists(check_path):
                    print(f"   ⏭️  [已存在-跳过] {file}")
                    self.stats['skipped'] += 1
                    continue

                print(f"   📄 [新文件-处理中] {file}")
                file_path = os.path.join(cat_path, file)
                file_ext = os.path.splitext(file)[1].lower()
                start_time = time.time()
                text = ""
                method = ""

                # --- 路由分发 ---
                if file_ext in ['.xlsx', '.xls']:
                    method = "Excel表格"
                    text = self.process_excel(file_path)
                elif file_ext == '.pdf':
                    mode = self.analyze_pdf(file_path)
                    if mode == "text":
                        method = "PDF提取"
                        text = self.process_text(file_path)
                    else:
                        method = "EasyOCR(GPU)"
                        text = self.process_ocr(file_path)
                # 【修改 2】新增 TXT 处理逻辑
                elif file_ext == '.txt':
                    method = "TXT直读"
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    except:
                        try:
                            with open(file_path, 'r', encoding='gbk') as f:
                                text = f.read()
                        except:
                            text = ""

                # --- 保存 ---
                if text and len(text.strip()) > 10:
                    chunks = self.chunk_text(text)
                    for i, chunk in enumerate(chunks):
                        with open(os.path.join(out_dir, f"{base_name}_part{i}.txt"), 'w', encoding='utf-8') as f:
                            f.write(f"文件名:{file}\n分类:{category}\n方式:{method}\n\n{chunk}")

                    print(f"      ✅ 成功 | {method} | 切片: {len(chunks)} | 耗时: {time.time() - start_time:.1f}s")
                    self.stats['success'] += 1
                else:
                    print(f"      ❌ 失败 | 内容过少或无法识别")
                    self.stats['failed'] += 1

                gc.collect()

        print(f"\n🎉 完成! 跳过:{self.stats['skipped']} | 成功:{self.stats['success']} | 失败:{self.stats['failed']}")

if __name__ == "__main__":
    UniversalFileProcessor().run()