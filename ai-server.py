import asyncio
import websockets
import json
import os
import struct
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from CosyVoice.cosyvoice.utils.file_utils import load_wav, logging
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from pydub import AudioSegment
from typing import Dict, Any

# 常量定义
ASR_TASK_SERVER_URL = os.getenv("ASR_TASK_SERVER_URL", "ws://127.0.0.1:8765")
SAMPLE_RATE = 16000

# 读取音频文件并转换为 PCM 格式
def read_audio_file(file_path: str) -> bytes:
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 转换为 16kHz, 单声道, 16位
    return audio.raw_data

# 生成二进制消息
def create_binary_message(session_id: str, pcm_data: bytes) -> bytes:
    session_id_bytes = session_id.encode()
    session_id_len = len(session_id_bytes)
    pcm_len = len(pcm_data)
    message = struct.pack('>I', session_id_len) + session_id_bytes + struct.pack('>I', pcm_len) + pcm_data
    return message

# 生成文本消息
def create_text_message(session_id: str, message_type: str, state: str = None, text: str = None, mode: str = None) -> str:
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

class ModelManager:
    def __init__(self):
        self.vad_model = None
        self.sense_model = None
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.cosyvoice_model = None

    def load_models(self) -> None:
        self.vad_model = AutoModel(
            model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            model_revision="v2.0.4",
            max_end_silence_time=240,
            speech_noise_thres=0.8,
            disable_update=True,
            disable_pbar=True,
            device="cuda",
        )

        self.sense_model = AutoModel(
            model="iic/SenseVoiceSmall",
            device="cuda",
            disable_update=True,
            disable_pbar=True,
        )

        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            "xiaozhi/Qwen2.5-Coder-7B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        self.qwen_tokenizer = AutoTokenizer.from_pretrained("xiaozhi/Qwen2.5-Coder-7B-Instruct")

        self.cosyvoice_model = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

class SenseVoiceWorker:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.audio_buffer = np.array([], dtype=np.float32)
        self.chunk_size_ms = 240  # VAD duration
        self.chunk_size = int(SAMPLE_RATE / 1000 * self.chunk_size_ms)
        self.audio_process_last_pos_ms = 0
        self.vad_cache = {}
        self.vad_last_pos_ms = -1
        self.vad_cached_segments = []
        self.fast_reply_checked = False
        self.listening = False
        self.content = ''

    def reset(self) -> None:
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_process_last_pos_ms = 0
        self.vad_cache = {}
        self.vad_last_pos_ms = -1
        self.vad_cached_segments = []
        self.fast_reply_checked = False
        self.listening = False
        self.content = ''

    def on_audio_frame(self, frame: bytes) -> None:
        frame_fp32 = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768
        self.audio_buffer = np.concatenate([self.audio_buffer, frame_fp32])
        current_duration = self.audio_buffer.shape[0] / 16

        if not self.listening:
            return

        if self.get_unprocessed_duration() < self.chunk_size_ms:
            return
        self.generate_vad_segments()

        if len(self.vad_cached_segments) == 0:
            self.truncate()
            return

        if self.vad_last_pos_ms == -1:  # Speech still going on
            return

        silence_duration = self.get_silence_duration()

        if not self.fast_reply_checked and silence_duration >= self.fast_reply_silence_duration:
            start_time = time.time()
            self.fast_reply_checked = True
            self.content = self.generate_text()
            if self.is_question():
                logging.info(f'Fast reply detected: {self.content} (time: {time.time() - start_time:.3f}s)')
                self.reply()
            return

        if silence_duration >= self.reply_silence_duration:
            start_time = time.time()
            self.content = self.generate_text()
            logging.info(f'Silence detected: {self.content} (time: {time.time() - start_time:.3f}s)')
            self.reply()
            return

        if current_duration >= self.max_audio_duration:
            start_time = time.time()
            self.content = self.generate_text()
            logging.info(f'Max audio duration reached: {self.content} (time: {time.time() - start_time:.3f}s)')
            self.reply()
            return

    def generate_vad_segments(self) -> None:
        beg = self.audio_process_last_pos_ms * 16
        end = beg + self.chunk_size
        chunk = self.audio_buffer[beg:end]
        self.audio_process_last_pos_ms += self.chunk_size_ms

        result = self.model_manager.vad_model.generate(input=chunk, cache=self.vad_cache, chunk_size=self.chunk_size_ms)
        if len(result[0]['value']) > 0:
            self.vad_cached_segments.extend(result[0]['value'])
            self.vad_last_pos_ms = self.vad_cached_segments[-1][1]
            if self.vad_last_pos_ms != -1:
                self.fast_reply_checked = False

    def generate_text(self) -> str:
        result = self.model_manager.sense_model.generate(input=self.audio_buffer, cache={}, language='zh', use_itn=True)
        return rich_transcription_postprocess(result[0]['text'])

    def get_unprocessed_duration(self) -> float:
        return self.audio_buffer.shape[0] / 16 - self.audio_process_last_pos_ms

    def get_silence_duration(self) -> float:
        if self.vad_last_pos_ms == -1:
            return 0
        return self.audio_buffer.shape[0] / 16 - self.vad_last_pos_ms

    def is_question(self) -> bool:
        match_tokens = ['吗', '嘛', '么', '呢', '吧', '啦', '？', '?', '拜拜', '再见', '晚安', '退下']
        last_part = self.content[-3:]
        for token in match_tokens:
            if token in last_part:
                return True
        return False

class QwenWorker:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def process(self, sense_text: str) -> str:
        messages = [
            {"role": "system", "content": "You are esp-bot, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": sense_text}
        ]
        text = self.model_manager.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.model_manager.qwen_tokenizer([text], return_tensors="pt").to(self.model_manager.qwen_model.device)
        generated_ids = self.model_manager.qwen_model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        qwen_response = self.model_manager.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return qwen_response

class CosyVoiceWorker:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def process(self, qwen_response: str) -> bytes:
        prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
        cosyvoice_result = self.model_manager.cosyvoice_model.inference_zero_shot(qwen_response, '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)
        audio_output = cosyvoice_result[0]['tts_speech']
        return audio_output

class AudioProcessingWorker:
    def __init__(self, wsapp: websockets.WebSocketServerProtocol, session_id: str, model_manager: ModelManager):
        self.wsapp = wsapp
        self.session_id = session_id
        self.model_manager = model_manager
        self.audio_buffer = np.array([], dtype=np.float32)
        self.content = ''
        self.sense_voice_worker = SenseVoiceWorker(model_manager)
        self.qwen_worker = QwenWorker(model_manager)
        self.cosy_voice_worker = CosyVoiceWorker(model_manager)

    def on_audio_frame(self, frame: bytes) -> None:
        self.sense_voice_worker.on_audio_frame(frame)
        if self.sense_voice_worker.content:
            # Step 2: Qwen
            qwen_response = self.qwen_worker.process(self.sense_voice_worker.content)

            # Step 3: CosyVoice
            audio_output = self.cosy_voice_worker.process(qwen_response)

            # Send audio output back to client
            binary_message = create_binary_message(self.session_id, audio_output)
            asyncio.run(self.wsapp.send(binary_message))

class AudioProcessingServer:
    def __init__(self):
        self.model_manager = ModelManager()
        self.workers: Dict[str, AudioProcessingWorker] = {}

    def initialize(self) -> None:
        self.model_manager.load_models()

    def get_worker(self, websocket: websockets.WebSocketServerProtocol, session_id: str) -> AudioProcessingWorker:
        if session_id not in self.workers:
            self.workers[session_id] = AudioProcessingWorker(websocket, session_id, self.model_manager)
        return self.workers[session_id]

    def parse_binary_message(self, websocket: websockets.WebSocketServerProtocol, message: bytes) -> None:
        session_id_len = int.from_bytes(message[:4], 'big')
        session_id = message[4:4 + session_id_len].decode()
        pcm_len = int.from_bytes(message[4 + session_id_len:8 + session_id_len], 'big')
        pcm = message[8 + session_id_len:8 + session_id_len + pcm_len]

        worker = self.get_worker(websocket, session_id)
        if worker is not None:
            worker.on_audio_frame(pcm)
        else:
            logging.warning(f'Unknown session_id: {session_id}')

    def parse_text_message(self, websocket: websockets.WebSocketServerProtocol, message: str) -> None:
        data = json.loads(message)
        session_id = data['session_id']

        if data['type'] == 'listen':
            worker = self.get_worker(websocket, session_id)
            state = data['state']
            if state == 'detect':
                worker.detect(data['text'])
            elif state == 'start':
                worker.start(data['mode'])
            elif state == 'stop':
                worker.stop()
            logging.info(f'Worker {session_id} started {state}')
        elif data['type'] == 'finish':
            if session_id in self.workers:
                del self.workers[session_id]
                logging.info(f'Worker {session_id} finished')
        else:
            logging.warning(f'Unknown message type: {data["type"]}')

    async def on_message(self, websocket: websockets.WebSocketServerProtocol, message: Any) -> None:
        try:
            if isinstance(message, bytes):
                self.parse_binary_message(websocket, message)
            else:
                self.parse_text_message(websocket, message)
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)

    async def on_connect(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        logging.info('Client connected.')
        try:
            async for message in websocket:
                await self.on_message(websocket, message)
        except websockets.ConnectionClosed:
            logging.info('Client disconnected.')

    async def run(self) -> None:
        logging.info('Starting Audio Processing Server...')
        async with websockets.serve(self.on_connect, "0.0.0.0", 8765):
            await asyncio.Future()  # run forever

if __name__ == "__main__":
    task_server = AudioProcessingServer()
    task_server.initialize()
    asyncio.run(task_server.run())