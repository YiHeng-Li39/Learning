import os
import json
import uuid
import webbrowser
import socket
import re
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context

# 尝试导入 QA 引擎
try:
    from qa_engine import QAEngine
except ImportError:
    QAEngine = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "high_quality_vector_db")
# 定义处理后的文件目录，用于读取文件列表
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_pdf")
SESSION_FILE = os.path.join(BASE_DIR, "chat_sessions.json")
# Excel 生成目录
GEN_DIR = os.path.join(BASE_DIR, "static", "generated")
os.makedirs(GEN_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static")

# ============ 1. 启动初始化 ============
engine = None
try:
    if QAEngine:
        print("🔄 [System] 正在启动... 准备连接本地知识库...")
        engine = QAEngine(VECTOR_DB_PATH)
        engine.initialize() # 可选
        print("✅ [System] 知识库连接成功！")
except Exception as e:
    print(f"⚠️ [Warning] 知识库连接失败: {e}")

# ============ 2. 会话管理与停止控制 ============
current_session_id = None
sessions = {}
active_streams = {}


def load_sessions_from_file():
    global sessions
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
        except:
            sessions = {}


def save_sessions_to_file():
    try:
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except:
        pass


def create_new_session():
    global current_session_id
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "title": "新对话",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": []
    }
    current_session_id = session_id
    save_sessions_to_file()
    return session_id


load_sessions_from_file()
if not sessions:
    create_new_session()
else:
    sorted_ids = sorted(sessions.keys(), key=lambda k: sessions[k]['timestamp'], reverse=True)
    current_session_id = sorted_ids[0]

# ============ 3. 前端 UI 模板 ============
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>南小开 Pro </title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
    window.MathJax = {
      tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']], processEscapes: true },
      svg: { fontCache: 'global' }
    };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        :root { --bg-body: #131314; --bg-sidebar: #1e1f20; --bg-surface: #1e1f20; --bg-user: #282a2c; --text-main: #e3e3e3; --text-sub: #c4c7c5; --hover: #333537; --accent: #a8c7fa; --green: #4caf50; --red: #d96570; }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Google Sans', 'Segoe UI', Roboto, sans-serif; background-color: var(--bg-body); color: var(--text-main); height: 100vh; display: flex; overflow: hidden; font-size: 14px; }
        .sidebar { width: 280px; background-color: #0e0e0e; display: flex; flex-direction: column; padding: 20px 16px; transition: all 0.3s ease; border-right: 1px solid #333; flex-shrink: 0; }
        .top-bar { display: flex; align-items: center; gap: 16px; padding: 0 12px 20px 12px; }
        .logo-text { font-size: 22px; color: var(--text-main); font-weight: 500; letter-spacing: -0.5px; }
        .new-chat-btn { background-color: #1a1a1c; color: #e3e3e3; border: 1px solid #333; border-radius: 20px; padding: 12px 16px; display: flex; align-items: center; gap: 12px; cursor: pointer; transition: 0.2s; font-size: 14px; font-weight: 500; margin-bottom: 20px; }
        .new-chat-btn:hover { background-color: #28292c; }

        /* 历史记录与文件列表的容器 */
        .list-container { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; padding-right: 5px; }
        .section-title { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; margin-bottom: 5px; padding-left: 10px; }

        .history-item, .file-item { padding: 10px 12px; border-radius: 8px; cursor: pointer; color: var(--text-sub); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 13px; display: flex; align-items: center; gap: 10px; transition: background 0.2s; }
        .history-item:hover, .file-item:hover { background-color: var(--hover); color: #fff; }
        .history-item.active { background-color: #004a77; color: #fff; font-weight: 500; }
        .file-item i { color: var(--accent); }

        .delete-btn { margin-left: auto; color: #666; font-size: 12px; padding: 4px; display: none; }
        .history-item:hover .delete-btn { display: block; } .delete-btn:hover { color: #d96570; }

        .kb-card { background-color: #1a1a1c; border-radius: 12px; padding: 12px; margin-top: auto; border: 1px solid #333; }
        .kb-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 12px; color: var(--text-sub); }
        .kb-status { display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: bold; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background-color: #666; }
        .dot.online { background-color: var(--green); box-shadow: 0 0 8px rgba(76, 175, 80, 0.4); }
        .kb-info { font-size: 11px; color: #888; font-family: monospace; }
        .bottom-settings { border-top: 1px solid #333; padding-top: 10px; margin-top: 10px;}
        .setting-input { width: 100%; background: transparent; border: none; color: #666; padding: 10px; font-size: 12px; outline: none; }

        .main { flex: 1; display: flex; flex-direction: column; position: relative; background-color: var(--bg-body); }
        .chat-container { flex: 1; overflow-y: auto; padding: 20px 15% 40px 15%; display: flex; flex-direction: column; gap: 30px; }
        .message { display: flex; gap: 16px; line-height: 1.6; font-size: 16px; animation: fadeIn 0.3s ease; }
        .message.user { justify-content: flex-end; }
        .user-bubble { background-color: var(--bg-surface); padding: 12px 20px; border-radius: 20px; max-width: 80%; color: var(--text-main); }
        .message.assistant { justify-content: flex-start; }
        .ai-icon { width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; border-radius: 50%; background: linear-gradient(135deg, #4285f4, #d96570); flex-shrink: 0; font-size: 14px; color: white; }

        /* 复制按钮样式 */
        .ai-wrapper { display: flex; flex-direction: column; max-width: 100%; width: 100%; }
        .ai-text { padding-top: 4px; overflow-x: auto; }
        .msg-actions { margin-top: 8px; display: flex; gap: 10px; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
        .message:hover .msg-actions { opacity: 1; pointer-events: auto; }
        .action-btn { background: transparent; border: 1px solid #444; color: #aaa; border-radius: 4px; padding: 4px 10px; font-size: 12px; cursor: pointer; display: flex; align-items: center; gap: 6px; transition: all 0.2s; }
        .action-btn:hover { background: #333; color: #fff; border-color: #666; }

        details.think, details.code { background: #161618; border: 1px solid #333; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }
        summary.head, summary.code-head { padding: 8px 12px; cursor: pointer; font-size: 12px; font-weight: 500; display: flex; align-items: center; gap: 6px; user-select: none; }
        summary.head { color: #888; }
        summary.code-head { color: #d96570; }
        summary.head:hover, summary.code-head:hover { background-color: #1e1e20; }
        .body, .code-body { padding: 12px; color: #c4c7c5; border-top: 1px solid #333; font-family: 'Consolas', monospace; font-size: 13px; white-space: pre-wrap; background-color: #0b0b0c; }

        details.refs { background: #0f1812; border: 1px solid #2e4a33; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }
        summary.ref-head { padding: 8px 12px; cursor: pointer; color: #4caf50; font-size: 12px; font-weight: 600; display: flex; align-items: center; gap: 6px; user-select: none; background: #132016; }
        summary.ref-head:hover { background-color: #1a2b1f; }
        .ref-body { padding: 12px; color: #c4c7c5; border-top: 1px solid #2e4a33; font-size: 13px; line-height: 1.5; }

        .input-wrapper { padding: 20px 15% 40px 15%; background-color: var(--bg-body); }
        .input-bar { background-color: var(--bg-surface); border-radius: 30px; padding: 8px 10px 8px 24px; display: flex; align-items: center; transition: 0.2s; }
        .input-bar:focus-within { background-color: #282a2c; }
        input { flex: 1; background: transparent; border: none; color: var(--text-main); font-size: 16px; outline: none; padding: 10px 0; }

        .send-btn { width: 40px; height: 40px; border: none; background: transparent; color: var(--text-main); border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; }
        .send-btn:hover { background-color: #333; }
        .send-btn.stop { background-color: #d96570; color: white; }
        .send-btn.stop:hover { background-color: #b04c56; }

        .welcome-screen { flex: 1; display: flex; flex-direction: column; justify-content: center; padding-left: 10px; }
        .welcome-title { font-size: 56px; font-weight: 500; background: linear-gradient(90deg, #4285f4, #9b72cb, #d96570); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; letter-spacing: -2px; }
        .welcome-sub { font-size: 24px; color: #444746; font-weight: 500; }
        mjx-container { font-size: 115% !important; margin: 0.5em 0; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: 0; } }
        ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="top-bar"><div class="logo-text">南小开</div></div>
        <button class="new-chat-btn" onclick="newChat()"><i class="fas fa-plus"></i> 新建对话</button>

        <div class="list-container">
            <div>
                <div class="section-title">历史记录</div>
                <div id="history-list"></div>
            </div>

            <div>
                <div class="section-title">📚 知识库文档 (点击解读)</div>
                <div id="file-list"></div>
            </div>
        </div>

        <div class="kb-card">
            <div class="kb-header"><i class="fas fa-database"></i> 知识库<div class="kb-status"><div id="kb-dot" class="dot"></div><span id="kb-text">连接中...</span></div></div>
            <div class="kb-info" id="kb-detail">...</div>
        </div>
        <div class="bottom-settings"><input type="password" id="api-key" class="setting-input" placeholder="Paste API Key here..."></div>
    </div>
    <div class="main">
        <div class="chat-container" id="chat-box">
            <div class="welcome-screen" id="welcome">
                <div class="welcome-title">你好，我是南小开，一个金融精算的智酷</div>
                <div class="welcome-sub">已连接本地精算数据库和文献资料库</div>
            </div>
        </div>
        <div class="input-wrapper">
            <div class="input-bar">
                <input type="text" id="user-input" autocomplete="off" placeholder="输入问题，或点击左侧文档进行解读..." onkeypress="if(event.keyCode==13) handleEnter()">
                <button class="send-btn" id="send-btn" onclick="handleAction()">
                    <i class="fas fa-paper-plane" id="btn-icon"></i>
                </button>
            </div>
        </div>
    </div>
    <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const welcomeScreen = document.getElementById('welcome');
        const historyList = document.getElementById('history-list');
        const fileList = document.getElementById('file-list');
        const apiKeyInput = document.getElementById('api-key');
        const sendBtn = document.getElementById('send-btn');
        const btnIcon = document.getElementById('btn-icon');

        let currentSessionId = null;
        let isGenerating = false; 

        window.onload = async function() {
            userInput.value = "";
            checkKbStatus();
            await loadHistoryList();
            await loadFileList(); // 加载文件列表
            await loadCurrentSession();
        };

        function checkKbStatus() {
             fetch('/kb_stats').then(r=>r.json()).then(data => {
                const dot = document.getElementById('kb-dot');
                const text = document.getElementById('kb-text');
                const detail = document.getElementById('kb-detail');
                if (data.status === 'online') { dot.classList.add('online'); text.innerText = '在线'; text.style.color = '#e3e3e3'; detail.innerText = `已索引: ${data.count} 条数据`; } 
                else { text.innerText = '离线'; text.style.color = '#d96570'; detail.innerText = '未连接或无数据'; }
             }).catch(e=>{});
        }

        async function loadHistoryList() {
            try {
                const res = await fetch('/get_history_list');
                const data = await res.json();
                historyList.innerHTML = '';
                if (data.list && data.list.length > 0) {
                    data.list.forEach(item => {
                        const div = document.createElement('div');
                        div.className = `history-item ${item.id === data.current_id ? 'active' : ''}`;
                        div.innerHTML = `<i class="far fa-message"></i><span style="flex:1; overflow:hidden; text-overflow:ellipsis;">${item.title}</span><i class="fas fa-times delete-btn" onclick="deleteSession(event, '${item.id}')" title="删除"></i>`;
                        div.onclick = (e) => { if(!e.target.classList.contains('delete-btn')) loadSession(item.id); };
                        historyList.appendChild(div);
                    });
                    currentSessionId = data.current_id;
                }
            } catch(e) {}
        }

        // 【核心修改】新增：加载文件列表并实现点击解读
        async function loadFileList() {
            try {
                const res = await fetch('/get_processed_files');
                const data = await res.json();
                fileList.innerHTML = '';
                if (data.files && data.files.length > 0) {
                    data.files.forEach(file => {
                        const div = document.createElement('div');
                        div.className = 'file-item';
                        div.innerHTML = `<i class="fas fa-file-alt"></i><span style="flex:1; overflow:hidden; text-overflow:ellipsis;">${file}</span>`;
                        div.title = "点击开始解读此文档";
                        div.onclick = () => { triggerInterpretation(file); };
                        fileList.appendChild(div);
                    });
                } else {
                    fileList.innerHTML = '<div style="padding:10px; color:#666; font-size:12px; font-style:italic;">暂无处理后的文档</div>';
                }
            } catch(e) {}
        }

        // 【核心修改】新增：触发解读逻辑
        function triggerInterpretation(filename) {
            userInput.value = `请深度解读文档：《${filename}》，包括核心观点、研究方法和结论。`;
            handleAction();
        }

        async function loadSession(sid) {
            const res = await fetch(`/load_session/${sid}`);
            const data = await res.json();
            currentSessionId = sid;
            chatBox.innerHTML = ''; 
            welcomeScreen.style.display = 'none';
            if (data.messages.length === 0) { chatBox.appendChild(welcomeScreen); welcomeScreen.style.display = 'flex'; } 
            else {
                data.messages.forEach(msg => { appendMessage(msg.role, msg.content, false); });
                chatBox.scrollTop = chatBox.scrollHeight;
            }
            loadHistoryList();
        }

        async function newChat() {
            if(isGenerating) await stopGeneration(); 
            await fetch('/new_chat', {method: 'POST'});
            await loadHistoryList();
            await loadCurrentSession();
        }

        async function loadCurrentSession() {
             const res = await fetch('/get_history_list');
             const data = await res.json();
             if(data.current_id) loadSession(data.current_id);
        }

        async function deleteSession(e, sid) {
            e.stopPropagation();
            if(!confirm("确定删除？")) return;
            const res = await fetch(`/delete_session/${sid}`, {method: 'POST'});
            const data = await res.json();
            await loadHistoryList();
            if (data.new_current_id && data.new_current_id !== sid) { await loadSession(data.new_current_id); }
        }

        function render(text, el) {
            let preprocessed = text;
            preprocessed = preprocessed.replace(/\\\\\[/g, '$$').replace(/\\\\\]/g, '$$');
            preprocessed = preprocessed.replace(/\\\\\\(/g, '$').replace(/\\\\\\)/g, '$');

            let html = preprocessed.replace(/<run_python>([\s\S]*?)<\/run_python>/g, 
                '<details class="code"><summary class="code-head"><i class="fas fa-cogs"></i> 正在构建表格数据...</summary><div class="code-body">$1</div></details>');

            html = html.replace(/<references>([\s\S]*?)<\/references>/g, '<details class="refs"><summary class="ref-head"><i class="fas fa-book-open"></i> 参考资料 (点击展开)</summary><div class="ref-body">$1</div></details>');
            html = html.replace(/<think>([\s\S]*?)<\/think>/g, '<details class="think"><summary class="head"><i class="fas fa-brain"></i> 深度思考过程 (点击展开)</summary><div class="body">$1</div></details>');
            el.innerHTML = marked.parse(html);
            if (window.MathJax) { MathJax.typesetPromise([el]).catch(function (err) { console.log(err); }); }
        }

        function copyContent(btn) {
            const wrapper = btn.closest('.ai-wrapper');
            const textDiv = wrapper.querySelector('.ai-text');
            const text = textDiv.innerText;
            navigator.clipboard.writeText(text).then(() => {
                const originalHtml = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-check"></i> 已复制';
                btn.style.color = '#4caf50';
                btn.style.borderColor = '#4caf50';
                setTimeout(() => {
                    btn.innerHTML = originalHtml;
                    btn.style.color = '';
                    btn.style.borderColor = '';
                }, 2000);
            });
        }

        function appendMessage(role, text, isStreaming=false) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;
            if (role === 'user') { 
                msgDiv.innerHTML = `<div class="user-bubble">${text}</div>`; 
                chatBox.appendChild(msgDiv); 
            } 
            else {
                msgDiv.innerHTML = `
                    <div class="ai-icon"><i class="fas fa-robot"></i></div>
                    <div class="ai-wrapper">
                        <div class="ai-text"></div>
                        <div class="msg-actions">
                            <button class="action-btn" onclick="copyContent(this)" title="复制内容">
                                <i class="far fa-copy"></i> 复制
                            </button>
                        </div>
                    </div>`;
                chatBox.appendChild(msgDiv);
                const textDiv = msgDiv.querySelector('.ai-text');
                if(!isStreaming) render(text, textDiv);
                return textDiv; 
            }
        }

        function handleEnter() { if (!isGenerating) send(); }

        function handleAction() {
            if (isGenerating) { stopGeneration(); } 
            else { send(); }
        }

        function setUIState(state) {
            if (state === 'generating') {
                isGenerating = true;
                sendBtn.classList.add('stop'); 
                btnIcon.className = "fas fa-stop"; 
            } else {
                isGenerating = false;
                sendBtn.classList.remove('stop');
                btnIcon.className = "fas fa-paper-plane"; 
            }
        }

        async function stopGeneration() {
            try { await fetch('/stop_chat', { method: 'POST' }); } catch(e) {}
            setUIState('idle');
        }

        async function send() {
            const text = userInput.value.trim();
            const apiKey = apiKeyInput.value.trim();
            if(!text) return;
            if(!apiKey) { alert("请输入 API Key"); apiKeyInput.focus(); return; }
            if(welcomeScreen) welcomeScreen.style.display = 'none';

            appendMessage('user', text);
            userInput.value = '';

            setUIState('generating');

            const aiTextDiv = appendMessage('assistant', '', true);
            aiTextDiv.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i>';
            chatBox.scrollTop = chatBox.scrollHeight;

            let fullText = ""; 
            try {
                const response = await fetch('/chat_stream', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message: text, api_key: apiKey}) });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                aiTextDiv.innerHTML = ''; 

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, {stream: true});
                    fullText += chunk;
                    render(fullText, aiTextDiv);
                    chatBox.scrollTop = chatBox.scrollHeight;
                }
                loadHistoryList();
            } catch(e) { 
                if (fullText.length === 0) aiTextDiv.innerHTML = "❌ 网络错误";
            } finally {
                setUIState('idle');
            }
        }
    </script>
</body>
</html>
"""


# ============ 4. 后端路由 ============

@app.route('/')
def home(): return render_template_string(HTML_TEMPLATE)


@app.route('/kb_stats')
def kb_stats():
    if engine and engine.collection:
        try:
            return jsonify({"status": "online", "count": engine.collection.count()})
        except:
            pass
    return jsonify({"status": "offline", "count": 0})


@app.route('/get_history_list')
def get_history_list():
    summary_list = []
    for sid, data in sessions.items():
        summary_list.append({"id": sid, "title": data.get("title", "新对话"), "timestamp": data.get("timestamp", "")})
    summary_list.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify({"list": summary_list, "current_id": current_session_id})


# 【核心修改】新增：获取已处理文件列表的接口
@app.route('/get_processed_files')
def get_processed_files():
    files_set = set()
    if os.path.exists(PROCESSED_DIR):
        for root, dirs, files in os.walk(PROCESSED_DIR):
            for file in files:
                if file.endswith('.txt') and "_part" in file:
                    # 从 "paper_name_part0.txt" 中提取 "paper_name"
                    base_name = file.split('_part')[0]
                    # 尝试还原原始扩展名（假设是pdf或xlsx，这里只显示文件名）
                    files_set.add(base_name)

    # 将 set 转为 list 并按名称排序，限制显示最新的 20 个
    sorted_files = sorted(list(files_set))[:20]
    return jsonify({"files": sorted_files})


@app.route('/load_session/<sid>')
def route_load_session(sid):
    global current_session_id
    if sid in sessions:
        current_session_id = sid
        return jsonify(sessions[sid])
    return jsonify({"messages": []})


@app.route('/new_chat', methods=['POST'])
def new_chat():
    create_new_session()
    return jsonify({"status": "ok", "id": current_session_id})


@app.route('/delete_session/<sid>', methods=['POST'])
def delete_session(sid):
    global current_session_id
    if sid in sessions:
        del sessions[sid]
        save_sessions_to_file()
        if current_session_id == sid:
            if sessions:
                sorted_ids = sorted(sessions.keys(), key=lambda k: sessions[k]['timestamp'], reverse=True)
                current_session_id = sorted_ids[0]
            else:
                create_new_session()
    return jsonify({"status": "ok", "new_current_id": current_session_id})


@app.route('/stop_chat', methods=['POST'])
def stop_chat():
    global current_session_id
    if current_session_id in active_streams:
        active_streams[current_session_id] = False
        print(f"🛑 用户请求停止: {current_session_id}")
    return jsonify({"status": "stopped"})


@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    global current_session_id
    data = request.json
    user_msg = data.get('message')
    api_key = data.get('api_key')

    if not engine: return jsonify({"error": "引擎未就绪"}), 500
    if not current_session_id: create_new_session()
    engine.set_api_key(api_key)

    active_streams[current_session_id] = True

    def generate():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        sessions[current_session_id]["messages"].append({"role": "user", "content": user_msg})
        if len(sessions[current_session_id]["messages"]) == 1:
            sessions[current_session_id]["title"] = user_msg[:10] + ("..." if len(user_msg) > 10 else "")
        save_sessions_to_file()

        history_for_engine = []
        for msg in sessions[current_session_id]["messages"]:
            content = msg['content']
            if msg['role'] == 'assistant':
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                content = re.sub(r'<references>.*?</references>', '', content, flags=re.DOTALL)
                content = content.strip()
            history_for_engine.append({"role": msg['role'], "content": content})

        full_response = ""

        for chunk in engine.chat_stream(user_msg, history=history_for_engine):
            if active_streams.get(current_session_id) == False:
                full_response += "\n\n(🛑 用户已停止生成)"
                yield "\n\n(🛑 用户已停止生成)"
                break

            full_response += chunk
            yield chunk

        if active_streams.get(current_session_id) != False:
            code_match = re.search(r'<run_python>(.*?)</run_python>', full_response, re.DOTALL)
            if code_match:
                try:
                    code_content = code_match.group(1)
                    exec_globals = {"pd": pd, "os": os}
                    exec(code_content, exec_globals)

                    files = os.listdir(GEN_DIR)
                    if files:
                        latest_file = max([os.path.join(GEN_DIR, f) for f in files], key=os.path.getctime)
                        filename = os.path.basename(latest_file)
                        download_link = f"\n\n**✅ 表格已生成：**\n[📂 点击下载 {filename}](/static/generated/{filename})"
                        full_response += download_link
                        yield download_link
                except Exception as e:
                    err = f"\n\n❌ 生成出错: {e}"
                    full_response += err
                    yield err

        sessions[current_session_id]["messages"].append({"role": "assistant", "content": full_response})
        sessions[current_session_id]["timestamp"] = timestamp
        save_sessions_to_file()

    return Response(stream_with_context(generate()), mimetype='text/plain')


def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM);
        s.connect(("8.8.8.8", 80));
        ip = s.getsockname()[
            0];
        s.close();
        return ip
    except:
        return "127.0.0.1"


if __name__ == '__main__':
    ip = get_ip()
    port = 5000
    print(f"\n🚀 南小开已启动!")
    print(f"👉 本地: http://127.0.0.1:{port}")
    print(f"👉 远程: http://{ip}:{port}\n")
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port)