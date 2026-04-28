#!/usr/bin/env python3

import queue
import threading
from datetime import datetime
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from perception_msgs.msg import PersonInfo, Transcript
from dnn_utils.emotion_recognition import (
    EmotionVideoRecognition,
    EmotionAudioRecognition,
    EmotionTextRecognition,
)
from cv_bridge import CvBridge
from PIL import Image as PILImage
from sensor_msgs.msg import Image # Should always after PILImage
import cv2
import dlib
from torchvision import transforms
from audio_utils_msgs.msg import AudioFrame, VoiceActivity
from transformers import AutoTokenizer
import time
import hbba_lite
import os


SUPPORTED_AUDIO_FORMAT = 'signed_16'
SUPPORTED_CHANNEL_COUNT = 1
SUPPORTED_SAMPLING_FREQUENCY = 16000
AUDIO_WINDOW_DURATION =64000
AUDIO_HOP_S = 4.0
AUDIO_HOP = int(AUDIO_HOP_S * SUPPORTED_SAMPLING_FREQUENCY) 
MAX_LEN    = 128
STRIDE     = 64   
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

emotion_dir = "/home/introlab/.ros/records"  
os.makedirs(emotion_dir, exist_ok=True)

class VideoEmotionProcessor:
    def __init__(self, inference_type: str, device: torch.device):
        self.device = device
        self.detector = dlib.get_frontal_face_detector()
        self.model = EmotionVideoRecognition(inference_type)
        self._bridge = CvBridge()
        self._transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
        self._last_analysis_time = 0
        self._last_label = None

    def _get_largest_face(self, faces):
        if len(faces) == 0:
            return None
        areas = [(face.right() - face.left()) * (face.bottom() - face.top()) for face in faces]
        return faces[np.argmax(areas)]
    
    def process(self, frame: np.ndarray):
        current_time = time.time()
        if current_time -self._last_analysis_time < 1:
            return frame, self._last_label, None, None, (0, 0)
        self._last_analysis_time = current_time

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb)
        face = self._get_largest_face(faces)
        if face:
            # Get face coordinates
            x, y = face.left(), face.top()
            w = face.right() - face.left()
            h = face.bottom() - face.top()
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
            # Extract face region
            roi_rgb = rgb[y:y+h, x:x+w]
            label = ''
            probs = 0
            if np.sum([roi_rgb]) != 0:
                roi_gray_pil = PILImage.fromarray(roi_rgb)
                roi = self._transform(roi_gray_pil) #ONNX 
                roi_tensor = roi.unsqueeze(0)
                with torch.no_grad():
                    logits_np, self.classes = self.model(roi_tensor)

                    #ONNX Runtime always returns NumPy.
                    logits_t = torch.from_numpy(logits_np).float()  
                    probs_t = torch.softmax(logits_t, dim=1) 

                    probs = probs_t.cpu().numpy().flatten() # (C,) numpy
                    pred_idx = probs_t.argmax(dim=1).item()
                    label = self.classes[pred_idx]

                if label != self._last_label:
                    self._last_label = label

                return frame, logits_np.flatten(), probs, label, (x, y)
        return frame, None, None, None, (0, 0)

class AudioEmotionProcessor:
    def __init__(self, inference_type: str, device: torch.device, audio_analysis_interval: int, is_voice: bool):
        self.device = device
        self.audio_analysis_interval = audio_analysis_interval
        self.recog_model = EmotionAudioRecognition(inference_type)
        self._audio_frames_lock = threading.Lock()
        self._audio_frames = []
        self._audio_analysis_count = 0
        self._is_voice = is_voice

    def process(self, msg: AudioFrame):
        if msg.format != SUPPORTED_AUDIO_FORMAT or \
        msg.channel_count != SUPPORTED_CHANNEL_COUNT or \
        msg.sampling_frequency != SUPPORTED_SAMPLING_FREQUENCY:
            self.get_logger().error(
                'Invalid audio frame (msg.format={}, msg.channel_count={}, msg.sampling_frequency={}})'
                .format(msg.format, msg.channel_count, msg.sampling_frequency)
            )
            return None

        # Append to the global sliding buffer (always) so we have context
        with self._audio_frames_lock:
            audio_frame = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / -np.iinfo(np.int16).min
            self._audio_frames.append(torch.from_numpy(audio_frame))
            # maintain ~4s of audio for features
            if (len(self._audio_frames) - 1) * audio_frame.shape[0] >= AUDIO_WINDOW_DURATION:
                self._audio_frames.pop(0)

        if self._is_voice:
            self._voice_samples_accum += audio_frame.shape[0]

        now = time.time()
        ready_by_samples = self._voice_samples_accum >= AUDIO_HOP      
        ready_by_time    = (now - self._last_audio_analyse_ts) >= (AUDIO_HOP / SUPPORTED_SAMPLING_FREQUENCY)  # 4.0 s

        if ready_by_samples and ready_by_time:
            res = self._analyse()
            if res is not None:
                self._voice_samples_accum -= AUDIO_HOP
                self._last_audio_analyse_ts = now
                return res
        return None

    def _analyse(self):
        audio_buffer = self._get_audio_buffer()
        if audio_buffer is None:
            return None

        if self._is_voice and audio_buffer.size()[0] >= AUDIO_WINDOW_DURATION:           
            # ONNX
            audio_window_buffer = audio_buffer[-AUDIO_WINDOW_DURATION:]
            audio_window_buffer_mono = self._to_mono_tensor(audio_window_buffer)

            if audio_window_buffer_mono.dim() > 1:
                audio_window_buffer_mono = audio_window_buffer_mono.squeeze()
                if audio_window_buffer_mono.dim() > 1:
                    audio_window_buffer_mono = audio_window_buffer_mono.mean(dim=-1)

            if audio_window_buffer_mono.dtype != torch.float32:
                audio_window_buffer_mono = audio_window_buffer_mono.to(torch.float32)

            if audio_window_buffer_mono.abs().max().item() > 1.5:
                audio_window_buffer_mono = audio_window_buffer_mono / 32768.0

            target_len = AUDIO_WINDOW_DURATION
            T = audio_window_buffer_mono.numel()

            if T < target_len:
                pad = target_len - T
                left = pad // 2
                right = pad - left
                audio_window_buffer_mono = torch.nn.functional.pad(
                    audio_window_buffer_mono, (left, right)
                )
            elif T > target_len:
                start = (T - target_len) // 2
                audio_window_buffer_mono = audio_window_buffer_mono[start:start + target_len]
           
            with torch.no_grad():
                logits, self.classes = self.recog_model(audio_window_buffer_mono)
                
                #ONNX Runtime always returns NumPy.
                logits_t = torch.from_numpy(logits).float() 
                label = self.classes[torch.argmax(logits_t, dim=1).item()]
                probs = torch.softmax(logits_t, dim=1).cpu().numpy().flatten()
            return logits_t.cpu().numpy().flatten(), probs, label
        return None

    def _get_audio_buffer(self):
        with self._audio_frames_lock:
            if not self._audio_frames:
                return None
            audio_buffer = torch.cat(self._audio_frames, dim=0)
        if audio_buffer.size()[0] > AUDIO_WINDOW_DURATION:
            return audio_buffer[-AUDIO_WINDOW_DURATION:]
        return audio_buffer
    
    def _to_mono_tensor(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        return waveform

class TextEmotionProcessor:
    def __init__(self, inference_type: str, device: torch.device):
        self.device = device
        self.model = EmotionTextRecognition(inference_type)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.classes = None

    def process(self, msg: Transcript):
        if not msg.text:
            return None
            
        enc = self.tokenizer(
            msg.text,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids_chunks      = enc['input_ids']
        attention_mask_chunks = enc['attention_mask']

        logits_list = []
        with torch.no_grad():
            for input_ids, attn in zip(input_ids_chunks, attention_mask_chunks):
                input_ids = input_ids.unsqueeze(0)
                attn      = attn.unsqueeze(0)
                logits, self.classes = self.model(input_ids, attn)
                logits_list.append(torch.tensor(logits))

        if not logits_list:
            return None
        all_logits = torch.cat(logits_list, dim=0)
        mean_logits = all_logits.mean(dim=0, keepdim=True)
        
        probs = torch.softmax(mean_logits, dim=1).cpu().numpy().flatten()
        pred_idx = int(mean_logits.argmax(dim=1))
        pred_label = self.classes[pred_idx]
        return mean_logits.cpu().numpy().flatten(), probs, pred_label

def _softmax(x, axis=-1, temp=1.0):
    x = x / max(temp, 1e-6)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=axis, keepdims=True), 1e-8, None)

class MoEFuserLogits:
    def __init__(
        self,
        class_count=7,
        alphas=None,
        taus=None,
        gate_weights=None,
        gate_temp=1.0,
    ):
        self.C = class_count
        self.alphas = alphas
        self.taus = taus
        self.gw = gate_weights
        self.gate_temp = float(gate_temp)

    def _gate_score(self, r, d, q, alpha, avail):
        return (
            self.gw.get("bias", 0.0) +
            self.gw.get("w_r", 1.4) * r +
            self.gw.get("w_d", 1.0) * d +
            self.gw.get("w_q", 1.2) * q +
            self.gw.get("w_alpha", 0.6) * np.log(max(alpha, 1e-6)) +
            self.gw.get("w_avail", 0.8) * (1.0 if avail else 0.0)
        )

    def _confidence_from_logits(self, logits):
        probs = torch.softmax(torch.tensor(logits), dim=0)
        top2_values = torch.topk(probs, 2).values
        margin = float(top2_values[0] - top2_values[1])
        return margin

    def fuse(self, now_ts, inputs):
        rows = []
        used = []
        for m in ("vision", "audio", "text"):
            info = inputs.get(m)
            if not info:
                continue
            avail = bool(info.get("available", False))
            if not avail:
                continue

            logits = info["logits"]
            t_upd = float(info.get("t_update", now_ts))
            age = max(0.0, now_ts - t_upd)
            tau = float(self.taus.get(m, 3.0))
            d = float(np.exp(-age / max(tau, 1e-6)))

            r = float(np.clip(info.get("reliability", 1.0), 0.0, 1.0))
            q = self._confidence_from_logits(logits)
            alpha = float(self.alphas.get(m, 1.0))
            
            score = self._gate_score(r=r, d=d, q=q, alpha=alpha, avail=avail)
            rows.append((m, logits, score))
            used.append(m)

        gates = {}
        if not rows:
            return None, None, gates, [], 0.0

        scores = np.array([row[2] for row in rows], dtype=np.float32)
        g = _softmax(scores, temp=self.gate_temp)
        
        fused_logits = np.zeros(self.C, dtype=np.float32)
        for (m, logits, _), gi in zip(rows, g):
            gates[m] = float(gi)
            fused_logits += float(gi) * logits

        fused_probs = torch.softmax(torch.tensor(fused_logits), dim=0).numpy()
        
        return fused_logits, fused_probs, gates, used, float(fused_probs.max())

class EmotionRecognitionNode(Node):
    def __init__(self):
        super().__init__('emotion_recognition_node')
        self.get_logger().info("Emotion Recognition Node started")

        try: 
            self._inference_type = self.declare_parameter('inference_type', 'torchscript').get_parameter_value().string_value
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._window_secs = self.declare_parameter('window_secs', 5).get_parameter_value().integer_value
            self._w_video = self.declare_parameter('weight_video', 0.2).get_parameter_value().double_value
            self._w_audio = self.declare_parameter('weight_audio', 0.1).get_parameter_value().double_value
            self._w_text = self.declare_parameter('weight_text', 0.7).get_parameter_value().double_value
            self._audio_analysis_interval = self.declare_parameter('interval', 64000).get_parameter_value().integer_value
            self.AVATA_LABELS = ["angry","disgust","fear","happy","normal","sad","blink"]
            self._emotion_classes = ['angry', 'disgust','fearful','happy','neutral','sad','surprised']
          
            self._executor = MultiThreadedExecutor(num_threads=4)
            self._bridge = CvBridge()
            self._emotion_classes = ['angry', 'disgust','fearful','happy','neutral','sad','surprised']
            self._is_voice = False

            # Publishers
            self._emotion_person_info_pub = self.create_publisher(PersonInfo, 'emotion_person_info', 5)
            
            # Subscriptions
            cbg = ReentrantCallbackGroup()
            self._image_sub = hbba_lite.ThrottlingHbbaSubscriber(self, 
                                                                 Image, 
                                                                 'camera_3d/color/image_raw',
                                                                 self._video_cb, 
                                                                 1, 
                                                                 "emotion_recognition/image_raw/filter_state"
                                                                )
            self._voice_activity_pub = self.create_subscription(VoiceActivity,
                                                                'voice_activity',
                                                                self._voice_activity_cb,
                                                                10)
            self._voice_sub = hbba_lite.OnOffHbbaSubscriber(self, 
                                                            AudioFrame,
                                                            'emotion_audio_in',
                                                            self._audio_cb,
                                                            10,
                                                            "emotion_recognition/emotion_audio_in/filter_state"
                                                            )
            self._voice_sub.on_filter_state_changed(self._filter_state_changed_cb)
            
            self._transcript_sub = self.create_subscription(
                                                    Transcript,
                                                    "speech_to_text/transcript",
                                                    self._text_cb,
                                                    1,
                                                    callback_group=cbg)

            self._video_processor = VideoEmotionProcessor(self._inference_type, self._device)
            self._audio_processor = AudioEmotionProcessor(self._inference_type, self._device, self._audio_analysis_interval, self._is_voice)
            self._text_processor = TextEmotionProcessor(self._inference_type, self._device)

            self._voice_frames = []
            self._voice_sequence_queue = queue.Queue()

            # Fusion state
            self._video_logits_buf = []
            self._audio_logits = None
            self._text_logits = None
            self._current_fused_label = ''
            self._fusion_timer = self.create_timer(self._window_secs, self._fusion_cb)
            self._fuser = MoEFuserLogits(
                alphas=dict(vision=0.7, audio=0.3),
                gate_weights=dict(
                    bias=0.0, w_r=1.0, w_d=0.5, w_q=1.5, w_alpha=0.8, w_avail=1.0 
                ),
                gate_temp=1.0,
            )
            
            self._save_emotion_log = False  # Set to True to enable logging of emotions with timestamps
        except Exception as e:
            self.get_logger().error(f'Error during initialization: {str(e)}')
            raise

    def _save_emotion_with_timestamp(self, text: str) -> None:
        file_name = os.path.join(emotion_dir, f"emotion_log.txt")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        with open(file_name, "a", encoding="utf-8") as f:
            f.write(f"{ts}: {text}\n")
        
        self.get_logger().info(f"Saved text: {ts}: {text}")

    def _video_cb(self, msg):
        try:
            if msg is not None:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                frame, logits, probs, label, label_position = self._video_processor.process(frame)

                if label is not None:
                    if self._save_emotion_log:
                        text = f"video = {label}"
                        self._save_emotion_with_timestamp(text)
                    if logits is not None:
                        self._video_logits_buf.append(logits)
        except Exception as e:
            self.get_logger().error(f"Error in _video_cb callback: {str(e)}")
            return
    
    def _voice_activity_cb(self, msg):
        last_is_voice = self._is_voice
        self._is_voice = msg.is_voice
        if last_is_voice and not self._is_voice:
            self._put_frames_in_voice_sequence_queue()
    
    def _filter_state_changed_cb(self, previous_is_filtering_all_messages, new_is_filtering_all_messages):
        if not previous_is_filtering_all_messages and new_is_filtering_all_messages:
            self.get_logger().info('_filter_state_changed_cb, put voice in frame')
            self._is_voice = False
            self._put_frames_in_voice_sequence_queue()

    def _put_frames_in_voice_sequence_queue(self):
        if len(self._voice_frames) > 0:
            self._voice_sequence_queue.put(np.concatenate(self._voice_frames))
            self._voice_frames.clear()

    def _audio_cb(self, msg):
        result = self._audio_processor.process(msg)
        if result is not None:
            self._audio_logits, probs, label = result
            if label is not None:
                if self._save_emotion_log:
                    text = f"audio = {label}"
                    self._save_emotion_with_timestamp(text)

    def _text_cb(self, msg: Transcript):
        if msg.text:
            self.get_logger().info(f" Text ={msg.text}")
        result = self._text_processor.process(msg)
        if result is None:
            self._text_logits = None
        else:
            self._text_logits, probs, label = result
            if label is not None:
                self.get_logger().info(f" text ={label} )")
                if self._save_emotion_log:
                    text = f"transcript = {label}"
                    self._save_emotion_with_timestamp(text)
    
    def _publish_emotion(self, label, pred_idx, conf, log_text):
        self.get_logger().info(log_text)
        self._current_fused_label = label
        
        person_info = PersonInfo()
        avata_idx = min(pred_idx, len(self.AVATA_LABELS) - 1)
        person_info.fused_avata_emotion_type = self.AVATA_LABELS[avata_idx]

        if self._save_emotion_log:
            self._save_emotion_with_timestamp(log_text)
            
        self._emotion_person_info_pub.publish(person_info)

    def _fusion_cb(self):
        now = time.time()

        # Text-Only Priority
        if self._text_logits is not None:
            try:
                logits = np.array(self._text_logits, dtype=np.float32).flatten()
                pred_idx = int(np.argmax(logits))
                label = self._emotion_classes[pred_idx]
                conf = float(_softmax(logits).max())

                log_txt = f"Fused emotion (TEXT ONLY): {label} (conf={conf*100:.1f}%)"
                self._publish_emotion(label, pred_idx, conf, log_txt)

            except Exception as e:
                self.get_logger().error(f"Text-only fusion error: {e}")

            # Reset
            self._text_logits = None
            self._video_logits_buf.clear()
            self._audio_logits = None
            return

        # Multimodal Fusion
        inputs = {}
        if self._video_logits_buf:
            inputs["vision"] = {
                "logits": np.mean(self._video_logits_buf, axis=0),
                "reliability": 0.6, "available": True, "t_update": now
            }

        if self._audio_logits is not None:
            inputs["audio"] = {
                "logits": self._audio_logits,
                "reliability": 0.2, "available": True, "t_update": now
            }

        if not inputs:
            return

        try:
            fused_logits, _, gates, used, conf = self._fuser.fuse(now, inputs)
            
            if fused_logits is not None:
                pred_idx = int(np.argmax(fused_logits))
                label = self._emotion_classes[pred_idx]
                
                log_txt = f"Fused emotion: {label} (conf={conf*100:.1f}%, gates={gates})"
                self._publish_emotion(label, pred_idx, conf, log_txt)

                # Reset
                if "vision" in used: self._video_logits_buf.clear()
                if "audio" in used: self._audio_logits = None
            else:
                self.get_logger().info("Fusion produced no logits.")
        except Exception as e:
            self.get_logger().error(f"Fusion error: {e}")

    def run(self):
        try:
            self._executor.add_node(self)
            self._executor.spin()
        finally: 
            self._voice_sequence_queue.put(None)

def main():
    rclpy.init()
    emotion_recogniton_node = EmotionRecognitionNode()
    try:
        emotion_recogniton_node.run()
    except KeyboardInterrupt:
        emotion_recogniton_node.get_logger().info("Emotion Recognition Node stopped by user")
    finally:
        emotion_recogniton_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()