
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

# 动态获取模型路径（已更换为更高灵敏度的 Jarvis 模型）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAKE_WORD_PATH = os.path.join(BASE_DIR, "models", "hey_jarvis_v0.1.tflite").replace("\\", "/")
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
        # 添加 60 秒超时限制，防止大脑卡死导致程序假死
        res = requests.post(OPENCLAW_API_URL, json={"model": "openclaw-vision", "messages": messages}, headers={"Content-Type": "application/json"}, timeout=60)
        
        # 🛑 拦截器：如果状态码不是 200 (成功)，直接把 OpenClaw 的真实报错打印出来
        if res.status_code != 200:
            print(f"\n❌ OpenClaw 拒绝了请求！状态码: {res.status_code}")
            print(f"📄 真实报错内容: {res.text}")
            speak("抱歉，我的大脑服务器返回了错误状态。")
            return
            
        try:
            # 尝试解析正常的 JSON 回复
            res_data = res.json()
            reply = res_data["choices"][0]["message"]["content"]
            speak(reply)
        except Exception as json_e:
            # 🛑 拦截器：如果是 200 但依然解析失败，打印出它到底传回了什么鬼东西
            print(f"\n❌ JSON 解析失败: {json_e}")
            print(f"📦 OpenClaw 传回的原始乱码是: {res.text}")
            speak("抱歉，大脑传回了无法解析的数据。")

    except requests.exceptions.ConnectionError:
        print("\n❌ 致命连接错误：目标计算机积极拒绝。请检查 OpenClaw 是否真的启动了，或者 18789 端口是否正确！")
        speak("抱歉，大脑中枢完全失联，请检查服务端是否启动。")
    except Exception as e:
        print(f"\n❌ 其他未知链接失败: {e}")
        speak("抱歉，链接大脑时发生了未知错误。")

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
            print(f"❌ 关键模型缺失: {p}\n请确保你已经下载了该模型并放入 models 文件夹！")
            return

    # 初始化唤醒模型 (提到外层，避免反复加载耗费内存)
    oww_model = Model(
        wakeword_models=[WAKE_WORD_PATH], 
        melspec_model_path=MELSPEC_PATH,
        embedding_model_path=EMBED_PATH,
        inference_framework="tflite"
    )

    # 外层循环：负责自适应扫描并锁定最优麦克风设备
    while True:
        P = pyaudio.PyAudio()
        target_device_index = None
        target_device_name = "Unknown"
        hw_channels = 1
        
        # 智能设备探针：遍历系统所有设备，优先抢占 USB 或外接麦克风
        for i in range(P.get_device_count()):
            info = P.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:
                name = info.get('name', '').lower()
                # 屏蔽电脑内部的虚拟混音通道
                if "mapper" in name or "mix" in name or "映射" in name or "混音" in name:
                    continue
                # 发现外接设备，立即锁定
                if "usb" in name or "external" in name:
                    target_device_index = i
                    target_device_name = info.get('name')
                    hw_channels = int(info.get('maxInputChannels'))
                    break
        
        # 如果没外接设备，使用系统默认的有效麦克风
        if target_device_index is None:
            try:
                default_info = P.get_default_input_device_info()
                target_device_index = default_info['index']
                target_device_name = default_info.get('name')
                hw_channels = int(default_info.get('maxInputChannels', 1))
            except Exception:
                print("❌ 未检测到麦克风，请插入设备或检查系统设置。5秒后重试...")
                time.sleep(5)
                P.terminate()
                continue

        print(f"\n🎤 智能路由已挂载设备: [{target_device_name}]")

        # 尝试激活音频流
        stream = None
        actual_rate = 16000
        for rate in [16000, 44100, 48000]:
            try:
                stream = P.open(format=pyaudio.paInt16, channels=hw_channels, rate=rate, input=True, input_device_index=target_device_index, frames_per_buffer=int(1280*(rate/16000)))
                actual_rate = rate
                print(f"✅ 底层神经流已激活 (采样率: {rate}Hz)")
                break
            except: continue

        if not stream:
            print("❌ 设备被其他程序独占，3秒后重试...")
            time.sleep(3)
            P.terminate()
            continue

        print(f"==============================================")
        print(f"✨ 终极系统就绪！请大声喊出：「Hey Jarvis」")
        print(f"==============================================")

        # 内层循环：持续捕捉音频数据进行高频检测
        try:
            while True:
                # exception_on_overflow=False 防止爆音引发程序崩溃
                raw_data = stream.read(int(1280*(actual_rate/16000)), exception_on_overflow=False)
                audio_data = np.frombuffer(raw_data, dtype=np.int16)
                
                # 多通道降维与降采样处理
                if hw_channels > 1: audio_data = audio_data[::hw_channels]
                if actual_rate != 16000: audio_data = audio_data[::(actual_rate // 16000)]
                
                prediction = oww_model.predict(audio_data)
                model_key = list(prediction.keys())[0]
                
                # 稍微降低判定阈值，提高灵敏度
                if prediction[model_key] > 0.15:
                    print(f"\n🚀 捕捉到唤醒词 [{model_key}]!")
                    
                    stream.stop_stream()
                    listen_after_wake()
                    stream.start_stream()
                    
                    # 【核心修复】：清理旧缓冲区，并重置唤醒模型的神经网络记忆
                    # 彻底解决二次唤醒困难、刚恢复就误触发的 Bug
                    time.sleep(0.2) 
                    if stream.get_read_available() > 0:
                        stream.read(stream.get_read_available(), exception_on_overflow=False)
                    
                    oww_model.reset()
                    print(f"\n✨ 记忆已重置，系统重新潜伏...")
                    
        except IOError:
            # 捕捉设备断开：如果你中途拔掉了 USB 麦克风，它会自动重启扫描寻找新设备
            print("\n⚠️ 麦克风物理连接断开，正在重新扫描系统硬件...")
            stream.stop_stream()
            stream.close()
            P.terminate()
            time.sleep(1)
            break # 退出内层监听，回到外层重新抓取麦克风

if __name__ == "__main__":
    try:
        start_wake_word_detection()
    except KeyboardInterrupt:
        print("\n系统已关")
