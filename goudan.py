
import os
import sys
import types
import io
import time
import requests
import base64
import shutil
import tempfile
import numpy as np
import pyaudio
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from io import BytesIO
from PIL import ImageGrab
from openwakeword.model import Model

# ================= 🛡️ 屏蔽 TensorFlow 冗余警告 =================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================= 🌉 加固版兼容桥梁 (欺骗 openwakeword) =================
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow as tf
        tr_mod = types.ModuleType("tflite_runtime")
        tri_mod = types.ModuleType("tflite_runtime.interpreter")
        tri_mod.Interpreter = tf.lite.Interpreter
        sys.modules["tflite_runtime"] = tr_mod
        sys.modules["tflite_runtime.interpreter"] = tri_mod
        print("🛠️ 兼容桥梁已加固：已将 TensorFlow 引擎成功映射至唤醒系统")
    except ImportError:
        print("❌ 严重错误：未检测到任何 AI 推理引擎，请运行: pip install tensorflow-cpu")
        sys.exit(1)

# ================= 🚨 配置区 (LBC 专属配置) 🚨 =================
# 建议定期更换 API Key 以保安全
SILICON_API_KEY = "sk-sipxazdnkxifppxcuhtnnwqlhqpskgkoxbpvssmwdecypyra"
OPENCLAW_API_URL = "http://127.0.0.1:18789/v1/chat/completions"
SOVITS_URL = "http://127.0.0.1:9880"

# 动态获取模型路径（解决路径中可能存在的中文或空格问题）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAKE_WORD_PATH = os.path.join(BASE_DIR, "models", "alexa_v0.1.tflite").replace("\\", "/")
MELSPEC_PATH = os.path.join(BASE_DIR, "models", "melspectrogram.tflite").replace("\\", "/")
EMBED_PATH = os.path.join(BASE_DIR, "models", "embedding_model.tflite").replace("\\", "/")

# ==========================================================

recognizer = sr.Recognizer()

def speak(text):
    print(f"\n🔊 发声: {text}")
    try:
        res = requests.get(f"{SOVITS_URL}?text={text}&text_language=zh", stream=True)
        if res.status_code == 200:
            data, fs = sf.read(io.BytesIO(res.content))
            sd.play(data, fs)
            sd.wait()
    except Exception as e:
        print(f"TTS 失败: {e}")

def ask_claw(text):
    print("🧠 大脑思考中...")
    needs_vision = any(keyword in text for keyword in ["看看", "屏幕", "截图", "报错", "这是什么"])
    messages = []
    
    if needs_vision:
        print("👁️ [视觉模式触发] 正在截取屏幕...")
        try:
            screenshot = ImageGrab.grab()
            buffered = BytesIO()
            screenshot.thumbnail((1280, 720)) 
            screenshot.save(buffered, format="JPEG", quality=80)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            messages = [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}]}]
        except Exception as e:
            messages = [{"role": "user", "content": text}] 
    else:
        messages = [{"role": "user", "content": text}]

    try:
        res = requests.post(OPENCLAW_API_URL, json={"model": "openclaw-vision", "messages": messages}, headers={"Content-Type": "application/json"})
        reply = res.json()["choices"][0]["message"]["content"]
        speak(reply)
    except Exception as e:
        print(f"OpenClaw 链接失败: {e}")

def listen_after_wake():
    # device_index=None 自动使用系统默认麦克风
    with sr.Microphone(device_index=None) as source:
        print("\n💡 [已唤醒] 我在听，请吩咐...")
        import winsound; winsound.Beep(1000, 200) 
        
        # 优化听觉门槛与环境噪音自适应
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            # 等待用户说话，超时时间设为 8 秒
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=12)
            print("☁️ 正在进行极速转录...")
            headers = {"Authorization": f"Bearer {SILICON_API_KEY}"}
            files = {"file": ("audio.wav", audio.get_wav_data(), "audio/wav")}
            res = requests.post("https://api.siliconflow.cn/v1/audio/transcriptions", headers=headers, files=files, data={"model": "FunAudioLLM/SenseVoiceSmall"})
            query = res.json().get("text", "").strip()
            if query:
                print(f"📝 你的指令: {query}")
                ask_claw(query)
        except Exception as e:
            print(f"聆听结束: {e}")

def start_wake_word_detection():
    # 🛡️ 模型文件存在性拦截
    for p in [WAKE_WORD_PATH, MELSPEC_PATH, EMBED_PATH]:
        if not os.path.exists(p):
            print(f"❌ 关键模型缺失: {p}")
            return

    P = pyaudio.PyAudio()
    # 自动获取当前 Windows 默认输入设备
    try:
        dev_info = P.get_default_input_device_info()
        print(f"🎤 自动挂载默认麦克风: {dev_info.get('name', 'Unknown')}")
    except Exception:
        print("❌ 找不到默认麦克风，请检查 Windows 声音设置。")
        return

    hw_channels = max(int(dev_info.get('maxInputChannels', 1)), 1)
    
    # 适配采样率启动流
    stream = None
    actual_rate = 16000
    for rate in [16000, 44100, 48000]:
        try:
            stream = P.open(format=pyaudio.paInt16, channels=hw_channels, rate=rate, input=True, input_device_index=None, frames_per_buffer=int(1280*(rate/16000)))
            actual_rate = rate
            print(f"✅ 麦克风已激活 (采样率: {rate}Hz)")
            break
        except: continue

    if not stream: return

    # 初始化唤醒模型，显式接管特征提取路径
    oww_model = Model(
        wakeword_models=[WAKE_WORD_PATH], 
        melspec_model_path=MELSPEC_PATH,
        embedding_model_path=EMBED_PATH,
        inference_framework="tflite"
    )
    
    print(f"\n==============================================")
    print(f"✨ 终极系统已就绪！请大声说出唤醒词...")
    print(f"==============================================")

    while True:
        try:
            raw_data = stream.read(int(1280*(actual_rate/16000)), exception_on_overflow=False)
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            # 音频降采样与通道合并
            if hw_channels > 1: audio_data = audio_data[::hw_channels]
            if actual_rate != 16000: audio_data = audio_data[::(actual_rate // 16000)]
            
            prediction = oww_model.predict(audio_data)
            # 获取模型名称键（防止路径差异导致 key 匹配失败）
            model_key = list(prediction.keys())[0]
            
            if prediction[model_key] > 0.2:
                print(f"\n🚀 捕捉到唤醒词 [{model_key}]!")
                # 【流控】：挂起监听，释放麦克风给录音模块
                stream.stop_stream()
                listen_after_wake()
                # 恢复监听
                stream.start_stream()
                print(f"\n✨ 重新进入监听状态...")
        except Exception:
            continue

if __name__ == "__main__":
    try:
        start_wake_word_detection()
    except KeyboardInterrupt:
        print("\n系统已关")
