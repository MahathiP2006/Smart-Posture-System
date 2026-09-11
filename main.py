import math
import time
import threading
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from plyer import notification
import av
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# Initialize MediaPipe Pose Solution
mp_pose = mp.solutions.pose

# Thread-safe global posture telemetry storage
posture_stats = {
    "good_count": 0,
    "bad_count": 0,
    "current_status": "Standby",
    "last_shoulder_roll": 0.0,
    "last_spine_angle": 0.0,
    "last_neck_angle": 0.0,
    "last_details": "Initializing vision pipeline...",
    "last_notification": 0.0,
    "lock": threading.Lock(),
}


def _normalized_to_pixel(normalized_x, normalized_y, width, height):
    if normalized_x is None or normalized_y is None:
        return None
    return int(normalized_x * width), int(normalized_y * height)


def _midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _angle_with_vertical(point_a, point_b):
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return abs(math.degrees(math.atan2(dx, dy)))


def _shoulder_roll_angle(left_shoulder, right_shoulder):
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    return abs(math.degrees(math.atan2(dy, dx)))


def analyze_posture(image, landmarks, tolerances):
    height, width = image.shape[:2]

    left_shoulder = _normalized_to_pixel(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
        width,
        height,
    )
    right_shoulder = _normalized_to_pixel(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
        width,
        height,
    )
    left_hip = _normalized_to_pixel(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
        width,
        height,
    )
    right_hip = _normalized_to_pixel(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
        width,
        height,
    )
    nose = _normalized_to_pixel(
        landmarks[mp_pose.PoseLandmark.NOSE].x,
        landmarks[mp_pose.PoseLandmark.NOSE].y,
        width,
        height,
    )
    left_ear = _normalized_to_pixel(
        landmarks[mp_pose.PoseLandmark.LEFT_EAR].x,
        landmarks[mp_pose.PoseLandmark.LEFT_EAR].y,
        width,
        height,
    )
    right_ear = _normalized_to_pixel(
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y,
        width,
        height,
    )

    if None in (left_shoulder, right_shoulder, left_hip, right_hip, nose):
        return (
            "No Detection",
            False,
            "Please step into camera view with upper body visible.",
            {},
        )

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    head_point = _midpoint(left_ear, right_ear) if (left_ear and right_ear) else nose

    shoulder_roll = _shoulder_roll_angle(left_shoulder, right_shoulder)
    spine_angle = _angle_with_vertical(shoulder_mid, hip_mid)
    neck_angle = _angle_with_vertical(head_point, shoulder_mid)

    messages = []
    if shoulder_roll > tolerances["shoulder_roll"]:
        messages.append(f"Shoulder Tilt ({shoulder_roll:.1f}°)")

    if spine_angle > tolerances["spine_angle"]:
        messages.append(f"Spine Deviation ({spine_angle:.1f}°)")

    if neck_angle > tolerances["neck_angle"]:
        messages.append(f"Forward Neck ({neck_angle:.1f}°)")

    is_good = len(messages) == 0
    status = "Good Posture" if is_good else "Bad Posture"
    details = "Optimal posture aligned!" if is_good else " | ".join(messages)

    return status, is_good, details, {
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "nose": nose,
        "head_point": head_point,
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
        "shoulder_roll": shoulder_roll,
        "spine_angle": spine_angle,
        "neck_angle": neck_angle,
    }


def draw_hud_overlay(image, posture_data, is_good, status, details):
    color = (46, 204, 113) if is_good else (231, 76, 60)  # Emerald Green vs Alizarin Red
    color_bgr = (46, 204, 113) if is_good else (60, 76, 231)

    left_shoulder = posture_data["left_shoulder"]
    right_shoulder = posture_data["right_shoulder"]
    left_hip = posture_data["left_hip"]
    right_hip = posture_data["right_hip"]
    shoulder_mid = posture_data["shoulder_mid"]
    hip_mid = posture_data["hip_mid"]
    head_point = posture_data["head_point"]

    # Draw skeleton lines with glow effect
    cv2.line(image, left_shoulder, right_shoulder, (255, 255, 255), 4)
    cv2.line(image, left_shoulder, right_shoulder, color_bgr, 2)

    cv2.line(image, left_shoulder, left_hip, (255, 255, 255), 4)
    cv2.line(image, left_shoulder, left_hip, color_bgr, 2)

    cv2.line(image, right_shoulder, right_hip, (255, 255, 255), 4)
    cv2.line(image, right_shoulder, right_hip, color_bgr, 2)

    s_mid_pt = (int(shoulder_mid[0]), int(shoulder_mid[1]))
    h_mid_pt = (int(hip_mid[0]), int(hip_mid[1]))
    head_pt = (int(head_point[0]), int(head_point[1]))

    cv2.line(image, s_mid_pt, h_mid_pt, (255, 255, 255), 4)
    cv2.line(image, s_mid_pt, h_mid_pt, color_bgr, 2)

    cv2.line(image, head_pt, s_mid_pt, (255, 255, 255), 3)
    cv2.line(image, head_pt, s_mid_pt, (0, 215, 255), 2)  # Gold neck vector

    # Draw joint nodes
    for pt in [left_shoulder, right_shoulder, left_hip, right_hip]:
        cv2.circle(image, pt, 7, (255, 255, 255), -1)
        cv2.circle(image, pt, 5, color_bgr, -1)

    cv2.circle(image, head_pt, 8, (255, 255, 255), -1)
    cv2.circle(image, head_pt, 6, (0, 215, 255), -1)

    # Upper HUD banner overlay
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (15, 15), (w - 15, 105), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    cv2.rectangle(image, (15, 15), (w - 15, 105), (51, 65, 85), 1)

    # Status Pill
    status_bg = (46, 204, 113) if is_good else (60, 76, 231)
    cv2.putText(
        image,
        f"SMART POSTURE: {status.upper()}",
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_bg,
        2,
        cv2.LINE_AA,
    )

    # Telemetry line
    cv2.putText(
        image,
        f"Shoulder Tilt: {posture_data['shoulder_roll']:.1f}° | Spine Vert: {posture_data['spine_angle']:.1f}° | Neck Tilt: {posture_data['neck_angle']:.1f}°",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (226, 232, 240),
        1,
        cv2.LINE_AA,
    )

    # Diagnostics detail line
    cv2.putText(
        image,
        details,
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )

    return image


class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.tolerances = {
            "shoulder_roll": 10.0,
            "spine_angle": 12.0,
            "neck_angle": 16.0,
        }
        self.notify_cooldown = 10.0

    def update_tolerances(self, shoulder, spine, neck, cooldown):
        self.tolerances["shoulder_roll"] = shoulder
        self.tolerances["spine_angle"] = spine
        self.tolerances["neck_angle"] = neck
        self.notify_cooldown = cooldown

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)

        if not results.pose_landmarks:
            h, w = image.shape[:2]
            cv2.rectangle(image, (20, 20), (w - 20, 70), (15, 23, 42), -1)
            cv2.putText(
                image,
                "SEARCHING FOR TARGET... Please align upper body in frame",
                (35, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (245, 158, 11),
                2,
                cv2.LINE_AA,
            )
            with posture_stats["lock"]:
                posture_stats["current_status"] = "No Detection"
                posture_stats["last_details"] = "Target missing from camera view"
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        status, is_good, details, posture_data = analyze_posture(
            image, results.pose_landmarks.landmark, self.tolerances
        )

        image = draw_hud_overlay(image, posture_data, is_good, status, details)

        with posture_stats["lock"]:
            posture_stats["current_status"] = status
            posture_stats["last_shoulder_roll"] = posture_data["shoulder_roll"]
            posture_stats["last_spine_angle"] = posture_data["spine_angle"]
            posture_stats["last_neck_angle"] = posture_data["neck_angle"]
            posture_stats["last_details"] = details

            if is_good:
                posture_stats["good_count"] += 1
            else:
                posture_stats["bad_count"] += 1
                now = time.monotonic()
                if now - posture_stats["last_notification"] > self.notify_cooldown:
                    try:
                        notification.notify(
                            title="Smart Posture Alert ⚠️",
                            message=f"Posture alert: {details}. Please sit upright and align your spine.",
                            timeout=3,
                        )
                    except Exception:
                        pass  # Fail gracefully if notification API is unsupported
                    posture_stats["last_notification"] = now

        return av.VideoFrame.from_ndarray(image, format="bgr24")


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Glassmorphic Container styling */
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            color: #f8fafc;
        }

        .stApp {
            background: #0f172a;
        }

        /* Hero Card Header */
        .hero-container {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #94a3b8;
            margin-top: 6px;
            font-weight: 500;
        }

        /* Metric Cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 12px;
            transition: all 0.2s ease-in-out;
        }

        .metric-card:hover {
            border-color: rgba(56, 189, 248, 0.3);
            transform: translateY(-2px);
        }

        .metric-label {
            font-size: 0.85rem;
            color: #94a3b8;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #f8fafc;
            margin-top: 4px;
        }

        .badge-good {
            display: inline-block;
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .badge-bad {
            display: inline-block;
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .badge-standby {
            display: inline-block;
            background: rgba(148, 163, 184, 0.15);
            color: #cbd5e1;
            border: 1px solid rgba(148, 163, 184, 0.3);
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }

        /* Custom Streamlit Buttons */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: white;
            border: none;
            padding: 8px 20px;
            transition: all 0.2s ease;
        }

        .stButton>button:hover {
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
            transform: scale(1.02);
        }
        </style>
        """,
        unsafe_allow_html=unsafe_allow_html,
    )


def reset_counters():
    with posture_stats["lock"]:
        posture_stats["good_count"] = 0
        posture_stats["bad_count"] = 0
        posture_stats["current_status"] = "Standby"


def main():
    st.set_page_config(
        page_title="Smart Posture System",
        page_icon="🧘‍♂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Hero Header Section
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">🧘‍♂️ Smart Posture System</div>
            <div class="hero-subtitle">Real-time Computer Vision Ergonomic Analysis & Health Monitoring powered by MediaPipe & WebRTC</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Ergonomic Settings
    with st.sidebar:
        st.header("⚙️ System Configuration")
        st.subheader("Ergonomic Tolerances")

        shoulder_tol = st.slider(
            "Shoulder Tilt Threshold (°)",
            min_value=5.0,
            max_value=20.0,
            value=10.0,
            step=1.0,
            help="Maximum acceptable shoulder roll inclination angle before triggering alert.",
        )

        spine_tol = st.slider(
            "Spine Deviation Threshold (°)",
            min_value=5.0,
            max_value=25.0,
            value=12.0,
            step=1.0,
            help="Maximum vertical spine deviation allowed.",
        )

        neck_tol = st.slider(
            "Neck Tilt Threshold (°)",
            min_value=5.0,
            max_value=30.0,
            value=16.0,
            step=1.0,
            help="Maximum forward head/neck tilt angle allowed.",
        )

        cooldown_sec = st.slider(
            "Notification Cooldown (s)",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
            help="Minimum interval between desktop posture reminder notifications.",
        )

        st.markdown("---")
        if st.button("🔄 Reset Telemetry Session", use_container_width=True):
            reset_counters()
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #64748b; text-align: center;">
                <b>Smart Posture System v2.0</b><br>
                Computer Vision & Deep Learning Lab
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Read current thread-safe posture metrics
    with posture_stats["lock"]:
        g_count = posture_stats["good_count"]
        b_count = posture_stats["bad_count"]
        c_status = posture_stats["current_status"]
        s_roll = posture_stats["last_shoulder_roll"]
        sp_angle = posture_stats["last_spine_angle"]
        n_angle = posture_stats["last_neck_angle"]
        dtls = posture_stats["last_details"]

    total_frames = g_count + b_count
    score_pct = int((g_count / total_frames * 100)) if total_frames > 0 else 100

    # Layout Columns
    col_video, col_stats = st.columns([1.5, 1])

    with col_video:
        st.subheader("🎥 Live Vision Stream & HUD")

        ctx = webrtc_streamer(
            key="smart-posture-streamer",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=PostureProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )

        if ctx.video_processor:
            ctx.video_processor.update_tolerances(
                shoulder_tol, spine_tol, neck_tol, cooldown_sec
            )

        st.info("💡 **Tip**: Sit ~2-3 feet from your webcam. Ensure your head, neck, and shoulders are clearly visible.")

    with col_stats:
        st.subheader("📊 Real-Time Posture Analytics")

        # Live Status Badge
        if c_status == "Good Posture":
            badge_html = '<span class="badge-good">🟢 GOOD POSTURE</span>'
        elif c_status == "Bad Posture":
            badge_html = '<span class="badge-bad">🔴 POSTURE ALERT</span>'
        else:
            badge_html = '<span class="badge-standby">⚪ STANDBY / SEARCHING</span>'

        st.markdown(f"**Current Status:** {badge_html}", unsafe_allow_html=True)

        st.write("")

        # Posture Score Progress Bar
        st.markdown(f"**Ergonomic Score:** **{score_pct}%**")
        st.progress(score_pct / 100.0)

        # Telemetry Cards Grid
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Good Frames</div>
                    <div class="metric-val" style="color: #4ade80;">{g_count:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Alert Frames</div>
                    <div class="metric-val" style="color: #f87171;">{b_count:,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Joint Angles Telemetry
        st.markdown("##### 📐 Joint Telemetry Angles")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.metric("Shoulder", f"{s_roll:.1f}°", delta=f"{s_roll - shoulder_tol:.1f}°" if s_roll > shoulder_tol else "Normal", delta_color="inverse")
        with t2:
            st.metric("Spine Vert", f"{sp_angle:.1f}°", delta=f"{sp_angle - spine_tol:.1f}°" if sp_angle > spine_tol else "Normal", delta_color="inverse")
        with t3:
            st.metric("Neck Tilt", f"{n_angle:.1f}°", delta=f"{n_angle - neck_tol:.1f}°" if n_angle > neck_tol else "Normal", delta_color="inverse")

        st.markdown("##### 🔍 Diagnostics")
        st.caption(f"_{dtls}_")


if __name__ == "__main__":
    main()