import asyncio
import websockets
import json
import os
import struct
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from pydub import AudioSegment

# 常量定义
ASR_TASK_SERVER_URL = os.getenv("ASR_TASK_SERVER_URL", "ws://127.0.0.1:8765")
SAMPLE_RATE = 16000

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

class ModelManager:
    def __init__(self):
        self.vad_model = None
        self.sense_model = None
        self.sv_model = None
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.cosyvoice_model = None

    def load_models(self):
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

class AsrWorker:
    def __init__(self, wsapp, session_id, model_manager):
        self.wsapp = wsapp
        self.session_id = session_id
        self.model_manager = model_manager
        self.audio_buffer = np.array([], dtype=np.float32)
        self.content = ''

    def on_audio_frame(self, frame):
        frame_fp32 = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768
        self.audio_buffer = np.concatenate([self.audio_buffer, frame_fp32])
        self.process_audio()

    def process_audio(self):
        # Step 1: SenseVoice
        sense_result = self.model_manager.sense_model.generate(input=self.audio_buffer, cache={}, language='zh', use_itn=True)
        sense_text = rich_transcription_postprocess(sense_result[0]['text'])

        # Step 2: Qwen
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

        # Step 3: CosyVoice
        prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
        cosyvoice_result = self.model_manager.cosyvoice_model.inference_zero_shot(qwen_response, '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)
        audio_output = cosyvoice_result[0]['tts_speech']

        # Send audio output back to client
        binary_message = create_binary_message(self.session_id, audio_output)
        asyncio.run(self.wsapp.send(binary_message))

class AsrTaskClient:
    def __init__(self):
        self.model_manager = ModelManager()
        self.workers = {}
        self.wsapp = None

    def initialize(self):
        self.model_manager.load_models()

    def get_worker(self, session_id):
        if session_id not in self.workers:
            self.workers[session_id] = AsrWorker(self.wsapp, session_id, self.model_manager)
        return self.workers[session_id]

    def parse_binary_message(self, message):
        session_id_len = int.from_bytes(message[:4], 'big')
        session_id = message[4:4 + session_id_len].decode()
        pcm_len = int.from_bytes(message[4 + session_id_len:8 + session_id_len], 'big')
        pcm = message[8 + session_id_len:8 + session_id_len + pcm_len]

        worker = self.get_worker(session_id)
        if worker is not None:
            worker.on_audio_frame(pcm)
        else:
            logging.warning(f'Unknown session_id: {session_id}')

    def parse_text_message(self, message):
        data = json.loads(message)
        session_id = data['session_id']

        if data['type'] == 'listen':
            worker = self.get_worker(session_id)
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

    async def on_message(self, wsapp, message):
        try:
            if isinstance(message, bytes):
                self.parse_binary_message(message)
            else:
                self.parse_text_message(message)
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)

    async def on_open(self, wsapp):
        logging.info('Connected to the Asr Task Server.')

    async def run(self):
        logging.info('Starting Asr Task Client...')
        self.wsapp = await websockets.connect(
            ASR_TASK_SERVER_URL,
            on_message=self.on_message,
            on_open=self.on_open
        )

        while True:
            try:
                await self.wsapp.run_forever()
            except Exception as e:
                logging.error(f"An error occurred: {e}", exc_info=True)
            # Remove all workers
            self.workers = {}
            logging.info('Reconnecting to the Asr Task Server in 3 seconds...')
            await asyncio.sleep(3)

if __name__ == "__main__":
    task_client = AsrTaskClient()
    task_client.initialize()
    asyncio.run(task_client.run())