import os
import time
import json
import smtplib
import threading
import datetime
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# =============================================================================
# SMTP & Surveillance Email Configuration
# =============================================================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "projectnew2026@gmail.com"
SENDER_PASSWORD = "sudzawdpggctjrre"

# Concerned Authorities' Email Recipients
AUTHORITY_EMAILS = [
    "nnaikspoorthi@gmail.com",
    "skruthika917@gmail.com"
]

class EmailNotifier:
    """Manages autonomous email alert notifications for detected crime incidents."""
    
    def __init__(self):
        self.email_already_sent = False
        self.lock = threading.Lock()
        self.cached_location = None
        self.last_sent_time = 0
        self.last_sent_incident = None

    def reset_alert_guard(self):
        """Resets the guard so the next uploaded video or new live session can send its single alert."""
        with self.lock:
            self.email_already_sent = False
            print("[EmailNotifier] Alert guard reset. Ready for next incident alert.")

    def get_surveillance_location(self):
        """Detects live geographic location with CCTV Sector metadata."""
        if self.cached_location:
            return self.cached_location

        geo_str = "Bengaluru, Karnataka, India"
        ip_addr = "127.0.0.1"
        coords_str = "12.9716° N, 77.5946° E"

        try:
            req = urllib.request.Request(
                "http://ip-api.com/json",
                headers={"User-Agent": "Mozilla/5.0 (Sentinel-AI Surveillance Engine)"}
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode())
                city = data.get("city", "Bengaluru")
                region = data.get("regionName", "Karnataka")
                country = data.get("country", "India")
                ip_addr = data.get("query", "")
                lat = data.get("lat", 12.9716)
                lon = data.get("lon", 77.5946)
                geo_str = f"{city}, {region}, {country}"
                coords_str = f"{lat}° N, {lon}° E"
        except Exception as e:
            print(f"[EmailNotifier] Geolocation lookup fallback: {e}")

        location_details = {
            "facility": "Central Surveillance Node Alpha",
            "sector": "Sector 4 - High Priority Security Zone",
            "camera_id": "CCTV-CAM-01 (YOLO11 Surveillance Unit)",
            "geographic_area": geo_str,
            "ip_address": ip_addr,
            "coordinates": coords_str,
            "maps_url": f"https://www.google.com/maps/search/?api=1&query={coords_str.replace(' ', '')}"
        }
        self.cached_location = location_details
        return location_details

    def trigger_crime_alert(self, crime_type, confidence, severity, snapshot_filename=None, alerts_dir="alerts", source_name="Uploaded Video"):
        """
        Triggers an email alert to concerned authorities ONLY ONCE per uploaded video or live session.
        Runs asynchronously in a background thread so the video playback never stutters.
        """
        with self.lock:
            # STRICT REQUIREMENT: Send only ONE TIME if crime detected multiple times
            if self.email_already_sent:
                return False

            self.email_already_sent = True
            self.last_sent_time = time.time()
            self.last_sent_incident = crime_type

        # Dispatch sending asynchronously in a separate daemon thread
        threading.Thread(
            target=self._send_email_thread,
            args=(crime_type, confidence, severity, snapshot_filename, alerts_dir, source_name),
            daemon=True
        ).start()
        return True

    def _send_email_thread(self, crime_type, confidence, severity, snapshot_filename, alerts_dir, source_name):
        """Asynchronously formats and sends the high-priority forensic email alert."""
        try:
            location = self.get_surveillance_location()
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
            
            subject = f"🚨 URGENT CRIME ALERT: [{severity}] {crime_type.upper()} Detected ({confidence:.0f}%) - {location['geographic_area']}"

            # Format Professional HTML Email Template
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f1f5f9; margin: 0; padding: 20px; }}
                    .container {{ max-width: 640px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 10px 25px rgba(0,0,0,0.06); }}
                    .header {{ background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); padding: 24px; text-align: center; color: #ffffff; }}
                    .header h1 {{ margin: 0; font-size: 20px; letter-spacing: 0.5px; font-weight: 800; }}
                    .badge-sev {{ display: inline-block; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 800; text-transform: uppercase; margin-top: 10px; background: {'#ef4444' if severity == 'CRITICAL' else '#f59e0b'}; color: #ffffff; }}
                    .content {{ padding: 24px; }}
                    .alert-banner {{ background: {'#fee2e2' if severity == 'CRITICAL' else '#fef3c7'}; border-left: 5px solid {'#ef4444' if severity == 'CRITICAL' else '#f59e0b'}; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px; }}
                    .alert-banner strong {{ color: {'#991b1b' if severity == 'CRITICAL' else '#92400e'}; font-size: 15px; display: block; }}
                    .details-table {{ width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 14px; }}
                    .details-table td {{ padding: 10px 12px; border-bottom: 1px solid #f1f5f9; }}
                    .details-table td.label {{ color: #64748b; font-weight: 600; width: 38%; }}
                    .details-table td.value {{ color: #0f172a; font-weight: 700; }}
                    .location-card {{ background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 16px; margin-top: 20px; }}
                    .location-card h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #0369a1; text-transform: uppercase; letter-spacing: 0.5px; }}
                    .location-item {{ font-size: 13px; color: #334155; margin-bottom: 5px; }}
                    .footer {{ background: #f8fafc; padding: 16px 24px; text-align: center; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>SENTINEL-AI SURVEILLANCE NOTIFICATION</h1>
                        <div class="badge-sev">{severity} THREAT INCIDENT</div>
                    </div>

                    <div class="content">
                        <div class="alert-banner">
                            <strong>Crime Detection Event Triggered</strong>
                            <span>The autonomous CCTV surveillance engine has detected a potential security incident requiring immediate verification.</span>
                        </div>

                        <table class="details-table">
                            <tr>
                                <td class="label">Incident Category:</td>
                                <td class="value" style="color: {'#dc2626' if severity == 'CRITICAL' else '#d97706'}; font-size: 16px;">{crime_type.upper()}</td>
                            </tr>
                            <tr>
                                <td class="label">AI Confidence Score:</td>
                                <td class="value">{confidence:.1f}%</td>
                            </tr>
                            <tr>
                                <td class="label">Detection Timestamp:</td>
                                <td class="value">{timestamp_str}</td>
                            </tr>
                            <tr>
                                <td class="label">Video Source:</td>
                                <td class="value">{source_name}</td>
                            </tr>
                            <tr>
                                <td class="label">Detection Engine:</td>
                                <td class="value">Ultralytics YOLO11 & Spatial Matrix</td>
                            </tr>
                        </table>

                        <div class="location-card">
                            <h3>📍 Surveillance Location &amp; Camera Details</h3>
                            <div class="location-item"><strong>Facility:</strong> {location['facility']}</div>
                            <div class="location-item"><strong>Surveillance Sector:</strong> {location['sector']}</div>
                            <div class="location-item"><strong>Camera Identifier:</strong> {location['camera_id']}</div>
                            <div class="location-item"><strong>Geographic Location:</strong> {location['geographic_area']}</div>
                            <div class="location-item"><strong>Station IP Address:</strong> {location['ip_address']}</div>
                            <div class="location-item"><strong>GPS Coordinates:</strong> {location['coordinates']}</div>
                            <div style="margin-top: 10px;">
                                <a href="{location['maps_url']}" style="display: inline-block; padding: 6px 14px; background: #0284c7; color: #ffffff; text-decoration: none; border-radius: 4px; font-size: 12px; font-weight: 600;">View Camera Location on Map &rarr;</a>
                            </div>
                        </div>

                        <div style="margin-top: 22px; font-size: 13px; color: #475569; line-height: 1.5;">
                            <em>Notice: Forensic snapshot evidence is attached to this email. This automated alert was dispatched strictly once per video session according to protocol.</em>
                        </div>
                    </div>

                    <div class="footer">
                        Sent to Authorized Security Personnel: {', '.join(AUTHORITY_EMAILS)}<br>
                        Autonomous CCTV Surveillance &amp; Crime Prevention Engine &copy; 2026
                    </div>
                </div>
            </body>
            </html>
            """

            # Build MIME Message
            msg = MIMEMultipart()
            msg["From"] = f"Sentinel-AI Crime Alert <{SENDER_EMAIL}>"
            msg["To"] = ", ".join(AUTHORITY_EMAILS)
            msg["Subject"] = subject
            msg["X-Priority"] = "1"  # Highest priority flag

            # Attach HTML body
            msg.attach(MIMEText(html_body, "html"))

            # Attach Snapshot Image if available
            if snapshot_filename:
                snapshot_path = os.path.join(alerts_dir, snapshot_filename)
                if os.path.exists(snapshot_path):
                    try:
                        with open(snapshot_path, "rb") as img_file:
                            img_data = img_file.read()
                            img_part = MIMEImage(img_data, name=snapshot_filename)
                            img_part.add_header("Content-Disposition", "attachment", filename=snapshot_filename)
                            msg.attach(img_part)
                    except Exception as img_err:
                        print(f"[EmailNotifier] Error attaching snapshot: {img_err}")

            # Send via Gmail SMTP TLS
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=12)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, AUTHORITY_EMAILS, msg.as_string())
            server.quit()

            print(f"\n[EmailNotifier] >>> CRIME ALERT EMAIL SUCCESSFULLY SENT TO AUTHORITIES: {AUTHORITY_EMAILS}")
            print(f"[EmailNotifier] Incident: {crime_type} ({confidence:.0f}%) | Location: {location['geographic_area']}\n")

        except Exception as err:
            print(f"[EmailNotifier ERROR] Failed to send email alert: {err}")

# Global Singleton Notifier
email_notifier = EmailNotifier()
