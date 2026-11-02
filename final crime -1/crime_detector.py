import os
import cv2
import time
import datetime
import numpy as np
import torch
from ultralytics import YOLO
from email_notifier import email_notifier

CRIME_CLASSES = [
    'Fighting',
    'Assault',
    'Robbery',
    'Stealing',
    'Shoplifting',
    'Burglary',
    'Shooting',
    'Arson',
    'Explosion',
    'Vandalism',
    'Normal activity'
]

class CrimeDetector:
    def __init__(self, models_dir="models", alerts_dir="alerts"):
        self.models_dir = models_dir
        self.alerts_dir = alerts_dir
        os.makedirs(self.alerts_dir, exist_ok=True)
        
        print("[CrimeDetector] Initializing Robust Multi-Crime Detection Engine...")
        self.violence_model = None
        self.weapon_model = None
        self.base_model = None
        self.pose_model = None

        # Load YOLO models
        v_path = os.path.join(models_dir, "violence_model.pt")
        if not os.path.exists(v_path):
            v_path = os.path.join(models_dir, "violence_yolov8n.pt")
        if os.path.exists(v_path):
            self.violence_model = YOLO(v_path)
            print("[CrimeDetector] Violence detection model loaded.")

        w_path = os.path.join(models_dir, "weapon_hadi.pt")
        if os.path.exists(w_path):
            self.weapon_model = YOLO(w_path)
            print("[CrimeDetector] Weapon model (pistol/knife) loaded.")

        # Load Base Ultralytics YOLO11 model
        y11_path = os.path.join(models_dir, "yolo11n.pt")
        if os.path.exists(y11_path):
            self.base_model = YOLO(y11_path)
            self.model_version = "YOLO11"
            print("[CrimeDetector] Base Ultralytics YOLO11 (yolo11n.pt) loaded successfully.")
        else:
            self.model_version = "YOLO11"

        # Load Ultralytics YOLO11 Pose model
        p11_path = os.path.join(models_dir, "yolo11n-pose.pt")
        if os.path.exists(p11_path):
            self.pose_model = YOLO(p11_path)
            print("[CrimeDetector] Ultralytics YOLO11 Pose model (yolo11n-pose.pt) loaded successfully.")

        # Configuration - Intrusion disabled by default to prevent webcam false alarms
        self.conf_threshold = 0.45
        self.enable_violence = True
        self.enable_weapons = True
        self.enable_intrusion = False  # Disabled by default so sitting in room doesn't trigger burglary!
        self.enable_fallen = True
        self.enable_abandoned_bag = False  # Disabled by default on webcam
        self.intrusion_zone = [0.55, 0.40, 0.98, 0.95]

        # 11-Class Real-time Probability State
        self.crime_classes = CRIME_CLASSES
        self.smoothed_probs = {c: (98.0 if c == 'Normal activity' else 0.2) for c in self.crime_classes}
        self.current_top_crime = "Normal activity"
        self.current_top_conf = 98.0

        # Motion & Interaction tracking
        self.prev_gray = None
        self.prev_mean_lum = None
        self.smoothed_motion = 0.0
        self.violence_streak = 0

        # Cooldown & Incidents
        self.last_alert_times = {}
        self.alert_cooldown_seconds = 4.0
        self.incidents = []

        # Telemetry
        self.current_threat_level = "SAFE"
        self.current_fps = 0.0
        self.active_detections = []
        self.frame_counter = 0
        self.last_process_time = time.time()
        self.video_analytics = []
        self.active_video_name = ""
        self.saved_upload_report = None
        self.persisted_incidents = []
        self.is_video_completed = False

    def set_config(self, conf=None, violence=None, weapons=None, intrusion=None, fallen=None, bags=None, zone=None):
        if conf is not None:
            self.conf_threshold = float(conf)
        if violence is not None:
            self.enable_violence = bool(violence)
        if weapons is not None:
            self.enable_weapons = bool(weapons)
        if intrusion is not None:
            self.enable_intrusion = bool(intrusion)
        if fallen is not None:
            self.enable_fallen = bool(fallen)
        if bags is not None:
            self.enable_abandoned_bag = bool(bags)
        if zone is not None and len(zone) == 4:
            self.intrusion_zone = [float(v) for v in zone]

    def reset_state(self, clear_incidents=False, clear_saved_report=False):
        """Resets detector telemetry to calm standby state without losing uploaded video report."""
        self.current_threat_level = "SAFE"
        self.smoothed_probs = {c: (98.0 if c == 'Normal activity' else 0.2) for c in self.crime_classes}
        self.current_top_crime = "Normal activity"
        self.current_top_conf = 98.0
        self.active_detections = []
        self.prev_gray = None
        self.prev_mean_lum = None
        self.smoothed_motion = 0.0
        self.violence_streak = 0
        self.current_fps = 0.0
        
        # Only clear saved report if explicitly requested (e.g. brand new file upload)
        if clear_saved_report:
            self.video_analytics.clear()
            self.active_video_name = ""
            self.saved_upload_report = None
            self.persisted_incidents.clear()
            self.is_video_completed = False
            email_notifier.reset_alert_guard()

        if clear_incidents:
            self.incidents.clear()

    def _trigger_incident(self, crime_type, confidence, bbox, frame):
        """Records an incident, saves snapshot, and triggers email alert to authorities (strictly once per session)."""
        now = time.time()
        last_time = self.last_alert_times.get(crime_type, 0)
        
        if now - last_time >= self.alert_cooldown_seconds:
            self.last_alert_times[crime_type] = now
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"alert_{crime_type.lower().replace(' ', '_')}_{file_ts}.jpg"
            filepath = os.path.join(self.alerts_dir, filename)
            
            try:
                cv2.imwrite(filepath, frame)
            except Exception as e:
                print(f"[CrimeDetector] Error saving snapshot: {e}")
                filename = ""

            critical_crimes = ['Fighting', 'Assault', 'Robbery', 'Shooting', 'Explosion', 'Arson']
            severity = "CRITICAL" if crime_type in critical_crimes else "WARNING"

            incident = {
                "id": len(self.incidents) + 1,
                "timestamp": timestamp_str,
                "type": crime_type,
                "label": f"{crime_type.upper()} DETECTED",
                "confidence": round(float(confidence), 1),
                "severity": severity,
                "snapshot": filename,
                "bbox": [int(b) for b in bbox] if len(bbox) == 4 else [0, 0, 0, 0]
            }
            self.incidents.insert(0, incident)
            self.persisted_incidents.insert(0, incident)
            if len(self.incidents) > 100:
                self.incidents.pop()

            # Autonomous Email Alert Notification (Strictly ONE TIME per uploaded video or live session)
            source_lbl = self.active_video_name if self.active_video_name else "Live Webcam Surveillance"
            email_notifier.trigger_crime_alert(
                crime_type=crime_type,
                confidence=float(confidence),
                severity=severity,
                snapshot_filename=filename,
                alerts_dir=self.alerts_dir,
                source_name=source_lbl
            )

    def _calculate_motion_energy(self, gray_frame):
        """Calculates pixel motion difference between consecutive frames with exponential smoothing."""
        if self.prev_gray is None or self.prev_gray.shape != gray_frame.shape:
            self.prev_gray = gray_frame.copy()
            self.smoothed_motion = 0.0
            return 0.0

        diff = cv2.absdiff(gray_frame, self.prev_gray)
        self.prev_gray = gray_frame.copy()
        raw_motion = float(np.mean(diff))
        self.smoothed_motion = 0.6 * raw_motion + 0.4 * self.smoothed_motion
        return self.smoothed_motion

    def _check_boxes_overlap(self, b1, b2):
        """Checks if two bounding boxes are in genuine physical contact/collision."""
        x1_max = max(b1[0], b2[0])
        y1_max = max(b1[1], b2[1])
        x2_min = min(b1[2], b2[2])
        y2_min = min(b1[3], b2[3])

        # If bounding boxes physically overlap in 2D space
        if x2_min > x1_max and y2_min > y1_max:
            return True

        # Centroid / edge proximity: must be very close (touching distance < 35% average body width)
        w1 = b1[2] - b1[0]
        w2 = b2[2] - b2[0]
        w_avg = (w1 + w2) / 2.0

        dx = max(0, b2[0] - b1[2]) if b2[0] > b1[2] else max(0, b1[0] - b2[2])
        dy = max(0, b2[1] - b1[3]) if b2[1] > b1[3] else max(0, b1[1] - b2[3])
        return dx < (w_avg * 0.35) and dy < (w_avg * 0.60)

    def process_frame(self, frame, skip_inference=False):
        """Processes a video frame with strict multi-person, motion, and pose verification."""
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        self.frame_counter += 1

        # FPS calculation
        now = time.time()
        elapsed = now - self.last_process_time
        self.last_process_time = now
        if elapsed > 0:
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / elapsed)

        # Fast path: On alternating frames for video playback speed, reuse cached detections
        if skip_inference and len(self.active_detections) > 0:
            return self._render_annotations(frame, self.active_detections, self.current_top_crime, self.current_top_conf, w, h)

        # Grayscale and motion energy
        small_gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        motion_energy = self._calculate_motion_energy(small_gray)

        # Baseline scores: Strong prior for Normal activity
        scores = {c: 0.01 for c in self.crime_classes}
        scores['Normal activity'] = 5.0

        detections = []
        best_box = [0, 0, w, h]

        # ---------------------------------------------------------------------
        # 1. Base YOLO Person & Weapon Detection
        # ---------------------------------------------------------------------
        persons = []
        weapons_detected = []
        theft_items = []

        if self.base_model:
            base_results = self.base_model.predict(frame, conf=0.35, verbose=False)
            for r in base_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cname = self.base_model.names[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()

                    if cname == 'person':
                        persons.append((xyxy, conf))
                    elif self.enable_weapons and cname in ['knife', 'baseball bat']:
                        if conf >= 0.62:
                            weapons_detected.append((cname, conf, xyxy))
                    elif cname in ['backpack', 'handbag', 'suitcase']:
                        if conf >= 0.45:
                            theft_items.append((xyxy, conf, cname))
                    elif cname == 'cell phone':
                        if conf >= 0.35:
                            theft_items.append((xyxy, conf, 'mobile phone'))
                    elif cname == 'laptop':
                        if conf >= 0.40:
                            theft_items.append((xyxy, conf, 'laptop'))

        # Check Specialized Weapons (Pistol / Knife)
        if self.enable_weapons and self.weapon_model:
            w_results = self.weapon_model.predict(frame, conf=0.52, verbose=False)
            for r in w_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cname = self.weapon_model.names[cls_id].lower()
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    if 'pistol' in cname or 'gun' in cname:
                        weapons_detected.append(('pistol', conf, xyxy))
                    elif 'knife' in cname and conf >= 0.58:
                        weapons_detected.append(('knife', conf, xyxy))

        num_persons = len(persons)

        # ---------------------------------------------------------------------
        # RULE: ANTI-FALSE-POSITIVE FOR SITTING / STANDING PERSONS
        # Two people standing in front of each other are NORMAL PERSONS!
        # Violence is ONLY triggered when there is actual physical struggle,
        # collision, active kinetic motion, and persistent contact.
        # ---------------------------------------------------------------------
        has_verified_fight = False

        if num_persons >= 2 and self.enable_violence and self.violence_model:
            # Check if any pair of people are in actual physical contact/proximity
            has_proximity = False
            for i in range(num_persons):
                for j in range(i + 1, num_persons):
                    if self._check_boxes_overlap(persons[i][0], persons[j][0]):
                        has_proximity = True
                        break
                if has_proximity:
                    break

            raw_violence_found = False
            v_box = None
            v_conf = 0.0

            # Only evaluate violence model if multiple people are in genuine proximity
            if has_proximity:
                v_results = self.violence_model.predict(frame, conf=0.65, verbose=False)
                for r in v_results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cname = self.violence_model.names[cls_id]
                        conf = float(box.conf[0])
                        if cname.lower() == 'violence' and conf >= 0.65:
                            raw_violence_found = True
                            v_box = box.xyxy[0].cpu().numpy()
                            v_conf = conf
                            break
                    if raw_violence_found:
                        break

            # Physical dynamic motion requirement:
            # Two men standing still facing each other (motion < 1.2) are NORMAL PERSONS!
            # Fighting only triggers if there is physical contact + active motion or extreme confidence!
            if raw_violence_found and has_proximity and (motion_energy >= 1.2 or v_conf >= 0.82):
                self.violence_streak += 1
            else:
                self.violence_streak = max(0, self.violence_streak - 1)

            # Require at least 2 consecutive positive frames
            if self.violence_streak >= 2 and v_box is not None:
                has_verified_fight = True
                scores['Fighting'] += 6.0 * v_conf
                scores['Assault'] += 4.5 * v_conf
                scores['Normal activity'] -= 5.0
                detections.append({
                    'type': 'Fighting',
                    'label': 'VIOLENCE / FIGHTING',
                    'conf': v_conf,
                    'box': v_box,
                    'color': (0, 0, 255),
                    'severity': 'CRITICAL'
                })
                best_box = v_box

        # ---------------------------------------------------------------------
        # If NO verified fight and NO weapons, all detected persons are NORMAL!
        # Two men standing facing each other will BOTH receive green NORMAL boxes!
        # ---------------------------------------------------------------------
        if not has_verified_fight and len(weapons_detected) == 0:
            for p_box, p_conf in persons:
                detections.append({
                    'type': 'Normal activity',
                    'label': 'PERSON (NORMAL)',
                    'conf': p_conf,
                    'box': p_box,
                    'color': (0, 230, 118),  # Calm emerald green
                    'severity': 'INFO'
                })

        # ---------------------------------------------------------------------
        # 2. Weapon Threat Processing
        # ---------------------------------------------------------------------
        for w_type, w_conf, w_box in weapons_detected:
            if 'pistol' in w_type or 'gun' in w_type:
                scores['Shooting'] += 5.5 * w_conf
                scores['Robbery'] += 4.5 * w_conf
                scores['Normal activity'] -= 4.5
                detections.append({
                    'type': 'Shooting',
                    'label': 'FIREARM / SHOOTING THREAT',
                    'conf': w_conf,
                    'box': w_box,
                    'color': (0, 0, 255),
                    'severity': 'CRITICAL'
                })
                best_box = w_box
            elif 'knife' in w_type:
                scores['Robbery'] += 4.5 * w_conf
                scores['Assault'] += 3.5 * w_conf
                scores['Normal activity'] -= 4.0
                detections.append({
                    'type': 'Robbery',
                    'label': 'WEAPON: KNIFE',
                    'conf': w_conf,
                    'box': w_box,
                    'color': (0, 0, 255),
                    'severity': 'CRITICAL'
                })
                best_box = w_box
            elif 'bat' in w_type:
                scores['Vandalism'] += 4.5 * w_conf
                scores['Assault'] += 3.0 * w_conf
                scores['Normal activity'] -= 3.5
                detections.append({
                    'type': 'Vandalism',
                    'label': 'WEAPON: BASEBALL BAT',
                    'conf': w_conf,
                    'box': w_box,
                    'color': (0, 0, 255),
                    'severity': 'CRITICAL'
                })
                best_box = w_box

        # ---------------------------------------------------------------------
        # 3. High-Value Target & Stealing / Shoplifting Processing
        # (Mobile Phones, Laptops, Handbags, Backpacks)
        # ---------------------------------------------------------------------
        for b_box, b_conf, b_name in theft_items:
            is_phone = ('phone' in b_name)
            # Differentiate Stealing (fast snatching motion or 2+ people grab) 
            # vs Shoplifting (1 person taking/handling/concealing item)
            is_snatching = (num_persons >= 2) or (motion_energy >= 1.4)

            if is_snatching:
                scores['Stealing'] += 6.5 * b_conf
                scores['Shoplifting'] += 4.0 * b_conf
                scores['Normal activity'] -= 5.0
                action_label = "THEFT / PHONE SNATCHING" if is_phone else f"THEFT / STEALING: {b_name.upper()}"
                c_type = 'Stealing'
                box_color = (0, 140, 255)  # Vivid Orange
            else:
                scores['Shoplifting'] += 6.5 * b_conf
                scores['Stealing'] += 4.5 * b_conf
                scores['Normal activity'] -= 5.0
                action_label = "SHOPLIFTING / CONCEALMENT" if is_phone else f"SHOPLIFTING TARGET: {b_name.upper()}"
                c_type = 'Shoplifting'
                box_color = (0, 190, 255)  # Amber Gold

            # If weapon also present, escalate to Robbery
            if len(weapons_detected) > 0:
                scores['Robbery'] += 4.0 * b_conf

            detections.append({
                'type': c_type,
                'label': f"{action_label}: {b_name.upper()}",
                'conf': b_conf,
                'box': b_box,
                'color': box_color,
                'severity': 'WARNING'
            })
            best_box = b_box

        # ---------------------------------------------------------------------
        # 4. Restricted Intrusion Zone (Burglary) - Only if explicitly enabled
        # ---------------------------------------------------------------------
        if self.enable_intrusion and num_persons > 0:
            zx1, zy1 = int(self.intrusion_zone[0]*w), int(self.intrusion_zone[1]*h)
            zx2, zy2 = int(self.intrusion_zone[2]*w), int(self.intrusion_zone[3]*h)
            for p_box, p_conf in persons:
                px = int((p_box[0] + p_box[2]) / 2)
                py = int(p_box[3])
                if zx1 <= px <= zx2 and zy1 <= py <= zy2:
                    scores['Burglary'] += 3.5 * p_conf
                    scores['Normal activity'] -= 2.5
                    detections.append({
                        'type': 'Burglary',
                        'label': 'BURGLARY / INTRUSION',
                        'conf': p_conf,
                        'box': p_box,
                        'color': (0, 140, 255),
                        'severity': 'WARNING'
                    })
                    best_box = p_box

        # ---------------------------------------------------------------------
        # 5. Fire / Arson Detection (Requires active motion/flicker + high intensity)
        # Prevents false alarms on static brown furniture or ambient room lighting
        # ---------------------------------------------------------------------
        if motion_energy > 4.5:
            hsv = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2HSV)
            lower_fire = np.array([0, 160, 220], dtype=np.uint8)
            upper_fire = np.array([22, 255, 255], dtype=np.uint8)
            fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
            fire_ratio = np.count_nonzero(fire_mask) / float(320 * 180)

            if fire_ratio > 0.035:  # Genuine large active flame
                fire_intensity = min(6.0, fire_ratio * 150.0)
                scores['Arson'] += fire_intensity
                scores['Explosion'] += fire_intensity * 0.6
                scores['Normal activity'] -= fire_intensity

                contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 200:
                        fx, fy, fw, fh = cv2.boundingRect(cnt)
                        scale_x, scale_y = w / 320.0, h / 180.0
                        box_orig = [fx * scale_x, fy * scale_y, (fx + fw) * scale_x, (fy + fh) * scale_y]
                        detections.append({
                            'type': 'Arson',
                            'label': 'ARSON / FLAME DETECTED',
                            'conf': min(0.98, fire_ratio * 15),
                            'box': box_orig,
                            'color': (0, 0, 255),
                            'severity': 'CRITICAL'
                        })
                        best_box = box_orig

        # ---------------------------------------------------------------------
        # 6. Normal Activity Guarantee when Calm / Sitting Alone without items
        # ---------------------------------------------------------------------
        if num_persons <= 1 and len(weapons_detected) == 0 and len(theft_items) == 0:
            # Single person sitting or walking with no weapons and no stolen/shoplifting items = 100% normal activity!
            scores['Normal activity'] = 9.0
            scores['Fighting'] = 0.01
            scores['Assault'] = 0.01
            scores['Burglary'] = 0.01
            scores['Robbery'] = 0.01
            scores['Shooting'] = 0.01
            scores['Arson'] = 0.01
            scores['Vandalism'] = 0.01
            scores['Stealing'] = 0.01
            scores['Shoplifting'] = 0.01

        # ---------------------------------------------------------------------
        # 6. Softmax Normalization over 11 Crime Categories
        # ---------------------------------------------------------------------
        score_vals = np.array([max(0.01, scores[c]) for c in self.crime_classes])
        exp_vals = np.exp(score_vals - np.max(score_vals))
        norm_probs = exp_vals / np.sum(exp_vals)

        alpha = 0.35
        for i, c in enumerate(self.crime_classes):
            current_pct = norm_probs[i] * 100.0
            self.smoothed_probs[c] = round(alpha * current_pct + (1.0 - alpha) * self.smoothed_probs[c], 1)

        top_crime = max(self.smoothed_probs, key=self.smoothed_probs.get)
        top_conf = self.smoothed_probs[top_crime]

        # Trigger incidents ONLY if genuine crime exceeds 45%
        if top_crime != 'Normal activity' and top_conf >= 45.0:
            critical_list = ['Fighting', 'Assault', 'Robbery', 'Shooting', 'Explosion', 'Arson']
            self.current_threat_level = "CRITICAL" if top_crime in critical_list else "ELEVATED"
            annotated_frame = self._render_annotations(frame, detections, top_crime, top_conf, w, h)
            self._trigger_incident(top_crime, top_conf, best_box, annotated_frame)
        else:
            self.current_threat_level = "SAFE"
            self.current_top_crime = "Normal activity"
            annotated_frame = self._render_annotations(frame, detections, "Normal activity", top_conf, w, h)

        self.current_top_crime = top_crime
        self.current_top_conf = top_conf
        self.active_detections = detections

        # Record real-time analytics data points for uploaded video distribution & area chart
        if self.active_video_name:
            time_sec = self.frame_counter / 30.0
            mins = int(time_sec // 60)
            secs = int(time_sec % 60)
            self.video_analytics.append({
                'time': f"{mins:02d}:{secs:02d}",
                'sec': round(time_sec, 1),
                'crime': top_crime,
                'conf': round(top_conf, 1),
                'threat': self.current_threat_level,
                'scores': {c: round(self.smoothed_probs[c], 1) for c in self.crime_classes}
            })
            # Continuously update persistent saved_upload_report
            self._update_saved_report(is_completed=False)

        return annotated_frame

    def _render_annotations(self, frame, detections, top_crime, top_conf, w, h):
        annotated = frame.copy()

        # ---------------------------------------------------------------------
        # 1. Responsive Top HUD Alert Banner (Auto-Scales & Auto-Centers)
        # ---------------------------------------------------------------------
        if top_crime != 'Normal activity' and top_conf >= 45.0:
            b_color = (0, 0, 200) if self.current_threat_level == "CRITICAL" else (0, 130, 230)
            banner_text = f"CRIME ALERT: {top_crime.upper()} ({top_conf:.0f}%)"
        else:
            b_color = (18, 45, 25)
            banner_text = f"SURVEILLANCE: NORMAL ACTIVITY ({top_conf:.0f}%)"

        # Dynamically scale font so text fits within width with padding
        target_max_w = w - 24
        font_scale_banner = 0.70
        (btw, bth), baseline = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_banner, 2)
        while btw > target_max_w and font_scale_banner > 0.35:
            font_scale_banner -= 0.04
            (btw, bth), baseline = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_banner, 2)

        banner_h = max(36, bth + baseline + 16)
        cv2.rectangle(annotated, (0, 0), (w, banner_h), b_color, -1)

        # Perfect horizontal and vertical centering
        bx = max(8, (w - btw) // 2)
        by = (banner_h + bth) // 2
        cv2.putText(annotated, banner_text, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_banner, (255, 255, 255), 2, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 2. Restricted Intrusion Zone (if enabled)
        # ---------------------------------------------------------------------
        if self.enable_intrusion:
            zx1, zy1 = int(self.intrusion_zone[0]*w), int(self.intrusion_zone[1]*h)
            zx2, zy2 = int(self.intrusion_zone[2]*w), int(self.intrusion_zone[3]*h)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 80, 220), -1)
            cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0, annotated)
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 140, 255), 2)
            cv2.putText(annotated, "[ RESTRICTED ZONE ]", (zx1 + 8, zy1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 140, 255), 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 3. Clean Bounding Boxes with Clamped & Wrapped Labels
        # ---------------------------------------------------------------------
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d['box']]
            color = d['color']
            label = f"{d['label']} {int(d['conf']*100)}%"
            severity = d['severity']

            thickness = 3 if severity == 'CRITICAL' else 2
            x1_draw = max(0, x1)
            y1_draw = max(banner_h, y1)
            x2_draw = min(w - 1, x2)
            y2_draw = min(h - 1, y2)
            cv2.rectangle(annotated, (x1_draw, y1_draw), (x2_draw, y2_draw), color, thickness)

            # Tactical corner brackets
            box_w = max(1, x2_draw - x1_draw)
            box_h = max(1, y2_draw - y1_draw)
            clen = min(20, max(6, int(box_w * 0.15)), box_w // 2, box_h // 2)
            cv2.line(annotated, (x1_draw, y1_draw), (x1_draw + clen, y1_draw), color, thickness + 2)
            cv2.line(annotated, (x1_draw, y1_draw), (x1_draw, y1_draw + clen), color, thickness + 2)
            cv2.line(annotated, (x2_draw, y1_draw), (x2_draw - clen, y1_draw), color, thickness + 2)
            cv2.line(annotated, (x2_draw, y1_draw), (x2_draw, y1_draw + clen), color, thickness + 2)
            cv2.line(annotated, (x1_draw, y2_draw), (x1_draw + clen, y2_draw), color, thickness + 2)
            cv2.line(annotated, (x1_draw, y2_draw), (x1_draw, y2_draw - clen), color, thickness + 2)
            cv2.line(annotated, (x2_draw, y2_draw), (x2_draw - clen, y2_draw), color, thickness + 2)
            cv2.line(annotated, (x2_draw, y2_draw), (x2_draw, y2_draw - clen), color, thickness + 2)

            # Auto-Scale and Auto-Clamp Label Pill so it NEVER cuts off
            font_scale_box = 0.48
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale_box, 1)
            while tw > (w - 20) and font_scale_box > 0.30:
                font_scale_box -= 0.04
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale_box, 1)

            pill_w = tw + 12
            pill_h = th + baseline + 8
            # Ensure pill stays within horizontal screen edges
            pill_x1 = max(4, min(x1, w - pill_w - 4))

            # If box is too close to top banner, render pill inside box
            if (y1 - pill_h - 4) < banner_h:
                pill_y1 = y1_draw + 4
            else:
                pill_y1 = y1_draw - pill_h - 4
            # Ensure pill stays within vertical screen bounds
            pill_y1 = max(banner_h + 2, min(pill_y1, h - pill_h - 4))
            text_y = pill_y1 + th + 3

            cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x1 + pill_w, pill_y1 + pill_h), color, -1)
            cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x1 + pill_w, pill_y1 + pill_h), (20, 20, 20), 1)
            txt_col = (255, 255, 255) if severity in ['CRITICAL', 'WARNING'] else (10, 10, 10)
            cv2.putText(annotated, label, (pill_x1 + 6, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_box, txt_col, 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 4. Responsive Telemetry Corner Overlay
        # ---------------------------------------------------------------------
        hud_bg = annotated.copy()
        hud_w = min(260, w - 20)
        hud_h = 38
        cv2.rectangle(hud_bg, (10, h - hud_h - 8), (10 + hud_w, h - 8), (15, 15, 15), -1)
        cv2.addWeighted(hud_bg, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (10, h - hud_h - 8), (10 + hud_w, h - 8), (55, 55, 55), 1)

        t_color = (0, 230, 118) if self.current_threat_level == "SAFE" else ((0, 165, 255) if self.current_threat_level == "ELEVATED" else (0, 50, 255))
        hud_font_scale = min(0.48, max(0.36, hud_w / 550.0))
        cv2.putText(annotated, f"THREAT: {self.current_threat_level}", (18, h - hud_h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, t_color, 1, cv2.LINE_AA)
        cv2.putText(annotated, f"FPS: {self.current_fps:.1f} | {top_crime} ({top_conf:.0f}%)", (18, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale - 0.04, (200, 200, 200), 1, cv2.LINE_AA)

        return annotated

    def get_status(self):
        """Returns live system telemetry, threat level, 11 crime category percentages, and email alert status."""
        loc = email_notifier.get_surveillance_location()
        return {
            "threat_level": self.current_threat_level,
            "model_version": getattr(self, "model_version", "YOLO11"),
            "fps": round(self.current_fps, 1),
            "top_crime": self.current_top_crime,
            "top_conf": self.current_top_conf,
            "crime_probabilities": self.smoothed_probs,
            "detection_count": len(self.active_detections),
            "incident_count": len(self.incidents),
            "email_alert_sent": email_notifier.email_already_sent,
            "email_recipients": ["vishwa.977@gmail.com", "vishwastudies7@gmail.com"],
            "surveillance_location": loc.get("geographic_area", "Surveillance Station"),
            "surveillance_facility": loc.get("facility", "Central Surveillance Node Alpha"),
            "config": {
                "confidence": self.conf_threshold,
                "violence": self.enable_violence,
                "weapons": self.enable_weapons,
                "intrusion": self.enable_intrusion,
                "fallen": self.enable_fallen,
                "abandoned_bag": self.enable_abandoned_bag,
                "intrusion_zone": self.intrusion_zone
            }
        }

    def _update_saved_report(self, is_completed=False):
        """Continuously aggregates and saves the uploaded video analysis report."""
        if not self.video_analytics:
            return

        # Calculate distribution across all frames in this uploaded video
        crime_counts = {}
        for pt in self.video_analytics:
            c = pt['crime']
            crime_counts[c] = crime_counts.get(c, 0) + 1

        total = max(1, len(self.video_analytics))
        distribution = {c: round((crime_counts.get(c, 0) / total) * 100, 1) for c in self.crime_classes}

        peak_conf = max((pt['conf'] for pt in self.video_analytics if pt['crime'] != 'Normal activity'), default=0)
        dominant_crime = max(crime_counts, key=crime_counts.get) if crime_counts else "Normal activity"

        # Determine overall threat level
        if any(pt['threat'] == 'CRITICAL' for pt in self.video_analytics) or peak_conf >= 55.0:
            overall_threat = "CRITICAL"
        elif any(pt['threat'] == 'ELEVATED' for pt in self.video_analytics) or peak_conf >= 40.0:
            overall_threat = "ELEVATED"
        else:
            overall_threat = "SAFE"

        # Sample timeline evenly so it covers the whole video from 00:00 to end
        step = max(1, len(self.video_analytics) // 75)
        timeline = self.video_analytics[::step]
        if self.video_analytics and (not timeline or timeline[-1] != self.video_analytics[-1]):
            timeline.append(self.video_analytics[-1])

        inc_list = list(self.persisted_incidents) if self.persisted_incidents else list(self.incidents)

        self.saved_upload_report = {
            "active_video": self.active_video_name or "Uploaded Video",
            "is_completed": is_completed,
            "timeline": timeline,
            "distribution": distribution,
            "kpis": {
                "total_points": len(self.video_analytics),
                "peak_conf": round(peak_conf, 1),
                "dominant_crime": dominant_crime,
                "threat_level": overall_threat,
                "incident_count": len(inc_list)
            },
            "incidents": inc_list
        }

    def finalize_upload_report(self):
        """Called when video finishes playing through once."""
        self.is_video_completed = True
        self._update_saved_report(is_completed=True)
        print(f"[CrimeDetector] Upload report finalized: {self.active_video_name} ({len(self.video_analytics)} frames analyzed)")

    def get_video_analytics(self):
        """Returns distribution and timeline area data for uploaded video analysis. NEVER resets to null."""
        if self.saved_upload_report is not None:
            return self.saved_upload_report

        return {
            "active_video": "Standby (Upload Video to Analyze)",
            "is_completed": False,
            "timeline": [],
            "distribution": {c: (100.0 if c == 'Normal activity' else 0.0) for c in self.crime_classes},
            "kpis": {
                "total_points": 0,
                "peak_conf": 0,
                "dominant_crime": "Normal activity",
                "threat_level": "SAFE",
                "incident_count": 0
            },
            "incidents": []
        }
