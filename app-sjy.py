# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
from threading import Thread
import torch
import streamlit as st

# 设置页面配置，修改布局为宽屏模式
st.set_page_config(page_title="Qwen2.5-Coder-7B-Instruct Chatbot", layout="wide")

# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    st.markdown("## Qwen2.5-Coder-7B-Instruct LLM")
    st.markdown("[算家云官网](https://www.suanjiayun.com/)")
    # 创建一个滑块，用于选择最大长度，范围在0到1024之间，默认值为512
    max_length = st.slider("max_length", 0, 4096, 1024, step=1)

# 创建一个标题和一个副标题
st.title("💬 Qwen2.5-Coder-7B-Instruct Chatbot")
st.caption("🚀 A streamlit chatbot powered by 算家云")

# 定义模型路径
model_name_or_path = 'Qwen2.5-Coder-7B-Instruct'

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    # 从预训练的模型中获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    # 从预训练的模型中获取模型，并设置模型参数
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
  
    return tokenizer, model

# 加载 Qwen2.5-Coder-7B-Instruct 的model和tokenizer
tokenizer, model = get_model()

# 如果session_state中没有"messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "有什么可以帮您的？"}]

# 遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 将用户的输入添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)
    
    # 构建输入     
    print(f"msg:{st.session_state.messages}")
    input_ids = tokenizer.apply_chat_template(st.session_state.messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt")['input_ids'].to('cuda')

    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_special_tokens=True, skip_prompt=True)
    kwargs = {'inputs': model_inputs, 'streamer': streamer, 'max_new_tokens': max_length}

    # Generation
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    # 在聊天界面上显示模型的输出
    with st.chat_message("assistant"):
        response = st.write_stream(streamer)
    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "assistant", "content": response})

# 添加自定义CSS，调整聊天区域布局
st.markdown("""
    <style>
        .streamlit-expanderHeader {
            font-size: 1.5rem;
            font-weight: bold;
        }

        .chat-container {
            width: 100%;
            max-width: 100%;
            margin: 0;
        }

        .stChatMessage {
            font-size: 18px;
            line-height: 1.5;
            padding: 10px;
        }

        .stTextArea {
            font-size: 16px;
            width: 100%;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

