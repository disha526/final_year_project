import cv2
import threading
import time
import os

class VideoManager:
    def __init__(self, detector):
        self.detector = detector
        self.cap = None
        self.source = None
        self.source_type = "none" # "webcam", "file", "sample", "url"
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.thread = None
        self.loop_video = True
        self.camera_index = 0

    def start_source(self, source, source_type="file", camera_index=0):
        """Switches or starts a new video source."""
        self.stop_source()
        
        with self.lock:
            self.source = source
            self.source_type = source_type
            self.camera_index = camera_index
            self.is_running = True
            
            if source_type == "webcam":
                print(f"[VideoManager] Opening webcam index {camera_index}...")
                self.cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
                self.loop_video = False
            else:
                print(f"[VideoManager] Opening video source: {source}...")
                self.cap = cv2.VideoCapture(source)
                # Sample videos loop, but uploaded user videos play once to generate complete prediction results
                self.loop_video = (source_type == "sample")

            if not self.cap or not self.cap.isOpened():
                print(f"[VideoManager] Failed to open source: {source}")
                self.is_running = False
                return False

            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True

    def stop_source(self):
        """Stops the current video capture thread and releases device immediately."""
        self.is_running = False
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.current_frame = None

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)

    def _capture_loop(self):
        """Continuously reads frames from source and processes with CrimeDetector."""
        fps = 30.0
        if self.cap and self.cap.isOpened():
            detected_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if detected_fps and detected_fps > 0 and detected_fps < 120:
                fps = detected_fps

        frame_duration = 1.0 / max(15.0, min(fps, 30.0))
        is_video_file = self.source_type in ["file", "sample", "upload"]
        playback_start_time = time.time()
        frame_counter = 0

        while self.is_running:
            start_t = time.time()
            if not self.cap or not self.cap.isOpened() or not self.is_running:
                break

            # Real-Time Wall-Clock Synchronization for Video Files
            # Eliminates slow-motion playback by syncing frame progress to real elapsed time
            if is_video_file:
                cur_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                target_pos = int((time.time() - playback_start_time) * fps)
                frames_behind = target_pos - cur_pos
                if frames_behind > 0:
                    for _ in range(min(frames_behind, 5)):
                        if not self.is_running or not self.cap.grab():
                            break

            if not self.is_running:
                break

            ret, frame = self.cap.read()
            if not ret or not self.is_running:
                if is_video_file and not self.loop_video and self.is_running:
                    print(f"[VideoManager] Uploaded video playback finished. Finalizing prediction results...")
                    if hasattr(self.detector, 'finalize_upload_report'):
                        self.detector.finalize_upload_report()
                    # Add completion banner to last frame
                    with self.lock:
                        if self.current_frame is not None and self.is_running:
                            h_cur, w_cur = self.current_frame.shape[:2]
                            cv2.rectangle(self.current_frame, (10, 10), (w_cur - 10, 48), (16, 185, 129), -1)
                            cv2.putText(self.current_frame, "[ PREDICTION COMPLETE - RESULTS PERMANENTLY SAVED ]", 
                                        (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1.0)
                    self.is_running = False
                    break
                elif self.loop_video and is_video_file and self.is_running:
                    # Rewind to start of video and reset sync clock
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    playback_start_time = time.time()
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue
                else:
                    break

            if not self.is_running:
                break

            # Resize if frame is huge (e.g. 4K or 1080p) to maintain high real-time FPS on CPU
            h, w = frame.shape[:2]
            if w > 960:
                scale = 960.0 / w
                frame = cv2.resize(frame, (960, int(h * scale)))

            # Run Crime Detection Pipeline with 1-in-3 inference cadence for real-time 1.0x playback
            frame_counter += 1
            skip_inference = is_video_file and (frame_counter % 3 != 0)
            processed_frame = self.detector.process_frame(frame, skip_inference=skip_inference)

            if not self.is_running:
                break

            with self.lock:
                if self.is_running:
                    self.current_frame = processed_frame

            elapsed = time.time() - start_t
            sleep_time = frame_duration - elapsed
            if sleep_time > 0 and self.is_running:
                time.sleep(sleep_time)

    def get_latest_frame_bytes(self):
        """Encodes the latest frame as JPEG bytes for MJPEG streaming."""
        with self.lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return None
        return buffer.tobytes()
