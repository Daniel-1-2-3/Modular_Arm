import base64
import json
import time
import argparse
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import threading
from color_detect_control import ColorDetectControl

SERVO_MIN = 0
SERVO_MAX = 270
MID_NEUTRAL = 15
END_NEUTRAL = 200

latest_frame = None
frame_lock = threading.Lock()


def on_message(client, userdata, message):
    global latest_frame
    if message.topic == "robot3/camera":
        try:
            data = json.loads(message.payload.decode())
            jpg_bytes = base64.b64decode(data["frame"])
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with frame_lock:
                latest_frame = rgb
        except Exception as e:
            print(f"[ERR] frame decode: {e}")


def clamp(val):
    return max(SERVO_MIN, min(SERVO_MAX, val))

def pub(client, topic, payload_dict):
    payload = json.dumps(payload_dict)
    client.publish(topic, payload)
    print(f"  → {topic}  {payload}")

def move_base(client, increment):
    pub(client, "robot1/motor", {"degrees": -increment})

def drive(client, increment):
    if increment < 0:
        direction = "backward"
    elif increment > 0:
        direction = "forward"
    else:
        direction = "stop"
    pub(client, "robot1/drive", {"speed": 40, "direction": direction})

def move_mid(client, curr, increment):
    new = curr - increment
    pub(client, "robot1/servo", {"angle": new})

def move_end(client, curr, increment):
    new = curr - increment
    pub(client, "robot2/motor", {"angle": new})

def reset(client):
    pub(client, "robot1/servo", {"angle": MID_NEUTRAL})
    pub(client, "robot2/motor", {"angle": END_NEUTRAL})
    return float(MID_NEUTRAL), float(END_NEUTRAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arm controller")
    parser.add_argument("--broker", default="192.168.40.65")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="arm_controller")
    client.on_message = on_message
    client.connect(args.broker, args.port, keepalive=60)
    client.subscribe("robot3/camera")
    client.loop_start()

    curr_mid = MID_NEUTRAL # 15
    curr_end = END_NEUTRAL # 200
    mid_angle, end_angle = reset(client)
    ctrl = ColorDetectControl()

    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        tof_reading = float("nan") # Inputs, but doesn't actualy use the tof
        action = ctrl.step(frame, tof_reading)
        print(f"  action - base:{action[0]:+.1f}  mid:{action[1]:+.1f}  end:{action[2]:+.1f}  rail:{action[3]:+.1f}")

        move_base(client, float(action[0]))
        drive(client, float(action[3]))

        move_end(client, curr_end, 0.5 * float(action[3]))
        curr_end += float(action[2])

        move_mid(client, curr_mid, 0.5 * float(action[3]))
        curr_mid += float(action[1])

        time.sleep(0.05)