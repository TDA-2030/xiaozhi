
import threading
import queue
# import oss2
import soundfile as sf
import base64
import os

class AudioStorage:
    def __init__(self, sample_rate):
        # self.oss_access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        # if self.oss_access_key_id is None:
        #     return
        # self.auth = oss2.Auth(self.oss_access_key_id, os.getenv("OSS_ACCESS_KEY_SECRET"))
        # self.bucket = oss2.Bucket(self.auth, os.getenv("OSS_ENDPOINT"), os.getenv("OSS_BUCKET_NAME"))
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.sample_rate = sample_rate
    
    def start(self):
        # if self.oss_access_key_id is None:
        #     return
        self.thread.start()
    
    def _run(self):
        print('AudioStorage thread started')
        while True:
            object_key, text, data = self.queue.get()
            self._store(object_key, text, data)
            self.queue.task_done()
    
    def _store(self, object_key, text, data):
        # use sf to get the wav file buffer from the audio buffer
        # from io import BytesIO
        with open(object_key, "wb") as f:
            sf.write(f, data, self.sample_rate)
            print(f'AudioStorage store text {text} to {object_key}')
            # f.seek(0)
            text64 = base64.b64encode(text.encode())
            # self.bucket.put_object(object_key, f.read(), headers={'x-oss-meta-text': text64})

    def async_put(self, object_key, text, data):
        # if self.oss_access_key_id is None:
        #     return
        self.queue.put((object_key, text, data))
