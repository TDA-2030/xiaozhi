import asyncio
import websockets
import json
import os
import struct
from pydub import AudioSegment

# 读取音频文件并转换为 PCM 格式
def read_audio_file(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 转换为 16kHz, 单声道, 16位
    return audio.raw_data

# 生成二进制消息
def create_binary_message(session_id, pcm_data):
    session_id_bytes = session_id.encode()
    session_id_len = len(session_id_bytes)
    pcm_len = len(pcm_data)
    message = struct.pack('>I', session_id_len) + session_id_bytes + struct.pack('>I', pcm_len) + pcm_data
    return message

# 生成文本消息
def create_text_message(session_id, message_type, state=None, text=None, mode=None):
    message = {
        'session_id': session_id,
        'type': message_type
    }
    if state:
        message['state'] = state
    if text:
        message['text'] = text
    if mode:
        message['mode'] = mode
    return json.dumps(message)

# 处理客户端连接
async def handle_client(websocket):
    session_id = "test_session"
    audio_file_path = "/home/zentek/work-zhouli/recording.wav"  # 替换为你的音频文件路径
    pcm_data = read_audio_file(audio_file_path)

    # 发送文本消息，通知客户端开始处理
    start_message = create_text_message(session_id, 'listen', state='start', mode='auto')
    await websocket.send(start_message)

    # 发送音频二进制数据
    binary_message = create_binary_message(session_id, pcm_data)
    await websocket.send(binary_message)

    # 发送文本消息，通知客户端停止处理
    stop_message = create_text_message(session_id, 'listen', state='stop')
    await websocket.send(stop_message)

    # 发送完成消息
    finish_message = create_text_message(session_id, 'finish')
    await websocket.send(finish_message)

    # 接收客户端的回复
    async for message in websocket:
        print(f"Received message from client: {message}")

# 启动 WebSocket 服务器
async def main():
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())