"""Desktop monitor and command client for sim_bridge.py.

Run from PowerShell:
    python src/bridge_client.py

For a tunneled bridge, pass its HTTPS base URL and set
MUJOCO_BRIDGE_TOKEN in the environment before starting the client.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from io import BytesIO
import json
import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any
from urllib.parse import urlparse

try:
    import websockets
except ImportError as error:
    raise SystemExit(
        "Missing client dependency. Install with: "
        "python -m pip install -r requirements-bridge.txt"
    ) from error

try:
    from PIL import Image, ImageTk
except ImportError as error:
    raise SystemExit(
        "Missing Pillow. Install with: "
        "python -m pip install -r requirements-bridge.txt"
    ) from error


CAMERA_NAMES = ("agentview", "topview")
JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "gripper_motor_joint",
)


def websocket_urls(base_url: str) -> tuple[str, str]:
    parsed = urlparse(base_url.rstrip("/"))
    if parsed.scheme not in ("http", "https", "ws", "wss") or not parsed.netloc:
        raise ValueError("Base URL must look like http://127.0.0.1:8765")
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}[
        parsed.scheme
    ]
    prefix = parsed.path.rstrip("/")
    return (
        f"{scheme}://{parsed.netloc}{prefix}/ws/feedback",
        f"{scheme}://{parsed.netloc}{prefix}/ws/commands",
    )


class BridgeClient:
    def __init__(self, base_url: str, token: str | None):
        self.feedback_url, self.commands_url = websocket_urls(base_url)
        self.token = token
        self.inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.outbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.sequence = 0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def send_target(self, targets: list[float]) -> int:
        self.sequence += 1
        seq = self.sequence
        self.outbox.put(
            {
                "type": "command",
                "seq": seq,
                "target_position_rad": targets,
            }
        )
        return seq

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._network_loop())
        except Exception as error:
            if not self.stop_event.is_set():
                self.inbox.put(("error", str(error)))

    async def _network_loop(self) -> None:
        async with (
            websockets.connect(self.feedback_url, max_size=32 * 1024 * 1024) as feedback,
            websockets.connect(self.commands_url, max_size=2 * 1024 * 1024) as commands,
        ):
            if self.token:
                auth = json.dumps({"token": self.token})
                await feedback.send(auth)
                await commands.send(auth)
                for socket in (feedback, commands):
                    response = json.loads(await socket.recv())
                    if response.get("type") != "authenticated":
                        raise RuntimeError(f"Bridge authentication failed: {response}")

            self.inbox.put(("status", "Connected to both bridge streams."))
            tasks = [
                asyncio.create_task(self._read_feedback(feedback)),
                asyncio.create_task(self._read_acks(commands)),
                asyncio.create_task(self._send_commands(commands)),
            ]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()

    async def _read_feedback(self, socket) -> None:
        async for payload in socket:
            self.inbox.put(("observation", json.loads(payload)))

    async def _read_acks(self, socket) -> None:
        async for payload in socket:
            self.inbox.put(("ack", json.loads(payload)))

    async def _send_commands(self, socket) -> None:
        while not self.stop_event.is_set():
            try:
                message = self.outbox.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            await socket.send(json.dumps(message))


class BridgeMonitor:
    def __init__(self, client: BridgeClient):
        self.client = client
        self.root = tk.Tk()
        self.root.title("MuJoCo Bridge Monitor")
        self.root.geometry("1200x820")
        self.root.minsize(900, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var = tk.StringVar(value="Connecting…")
        ttk.Label(self.root, textvariable=self.status_var).pack(
            anchor="w", padx=12, pady=(10, 4)
        )

        image_row = ttk.Frame(self.root)
        image_row.pack(fill="both", expand=True, padx=8, pady=8)
        self.image_labels: dict[str, ttk.Label] = {}
        self.photo_refs: dict[str, ImageTk.PhotoImage] = {}
        for name in CAMERA_NAMES:
            panel = ttk.LabelFrame(image_row, text=name)
            panel.pack(side="left", fill="both", expand=True, padx=4)
            label = ttk.Label(panel, text="Waiting for camera frame…", anchor="center")
            label.pack(fill="both", expand=True, padx=4, pady=4)
            self.image_labels[name] = label

        self.state_var = tk.StringVar(value="Waiting for state…")
        ttk.Label(
            self.root,
            textvariable=self.state_var,
            justify="left",
            font=("Consolas", 10),
        ).pack(anchor="w", padx=12, pady=6)

        control = ttk.LabelFrame(self.root, text="Target joint positions (radians)")
        control.pack(fill="x", padx=12, pady=(4, 10))
        self.target_vars: list[tk.StringVar] = []
        self.entries: list[ttk.Entry] = []
        self.limit_labels: list[ttk.Label] = []
        self.slew_labels: list[ttk.Label] = []
        for index, joint_name in enumerate(JOINT_NAMES):
            ttk.Label(control, text=joint_name).grid(
                row=0, column=index, padx=5, pady=(6, 2)
            )
            limit_label = ttk.Label(control, text="limit: waiting…")
            limit_label.grid(row=1, column=index, padx=5, pady=(0, 1))
            slew_label = ttk.Label(control, text="target rate: waiting…")
            slew_label.grid(row=2, column=index, padx=5, pady=(0, 2))
            value = tk.StringVar(value="")
            entry = ttk.Entry(control, width=12, textvariable=value)
            entry.grid(row=3, column=index, padx=5, pady=(0, 8))
            self.target_vars.append(value)
            self.entries.append(entry)
            self.limit_labels.append(limit_label)
            self.slew_labels.append(slew_label)

        button_row = ttk.Frame(control)
        button_row.grid(row=4, column=0, columnspan=len(JOINT_NAMES), pady=(0, 8))
        ttk.Button(button_row, text="Send target", command=self.send_target).pack(
            side="left", padx=5
        )
        ttk.Button(button_row, text="Hold current pose", command=self.hold_pose).pack(
            side="left", padx=5
        )

        self.latest_observation: dict[str, Any] | None = None
        self.fields_initialized = False
        self.client.start()
        self.root.after(30, self.poll_inbox)

    def poll_inbox(self) -> None:
        try:
            while True:
                event, payload = self.client.inbox.get_nowait()
                if event == "status":
                    self.status_var.set(payload)
                elif event == "error":
                    self.status_var.set(f"Connection error: {payload}")
                elif event == "ack":
                    if payload.get("accepted"):
                        self.status_var.set(
                            f"Command {payload.get('seq')} accepted; target is ramped by the simulator."
                        )
                    else:
                        self.status_var.set(
                            f"Command rejected: {payload.get('error', 'unknown error')}"
                        )
                elif event == "observation":
                    self.show_observation(payload)
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(30, self.poll_inbox)

    def show_observation(self, observation: dict[str, Any]) -> None:
        self.latest_observation = observation
        state = observation.get("state", {})
        limits = observation.get("joint_limits_rad", {})
        rates = observation.get("target_slew_rad_s", {})
        positions = state.get("joint_position_rad", [])
        velocities = state.get("joint_velocity_rad_s", [])
        commanded = state.get("commanded_position_rad", [])
        self.state_var.set(
            f"sim time: {observation.get('sim_time', 0.0):.3f} s    "
            f"observation: {observation.get('seq')}\n"
            f"position (rad):  {self._format_values(positions)}\n"
            f"velocity (rad/s): {self._format_values(velocities)}\n"
            f"command target:   {self._format_values(commanded)}"
        )

        if not self.fields_initialized and len(positions) == len(self.target_vars):
            for variable, position in zip(self.target_vars, positions):
                variable.set(f"{float(position):.5f}")
            self.fields_initialized = True

        for index, name in enumerate(JOINT_NAMES):
            bounds = limits.get(name)
            if bounds and len(bounds) == 2:
                self.limit_labels[index].configure(
                    text=f"limit: {float(bounds[0]):.2f} to {float(bounds[1]):.2f} rad"
                )
            rate = rates.get(name)
            if rate is not None:
                self.slew_labels[index].configure(
                    text=f"target slew ≤ {float(rate):.2f} rad/s"
                )

        for name in CAMERA_NAMES:
            frame = observation.get("images", {}).get(name)
            if not frame or not frame.get("data_base64"):
                continue
            raw = base64.b64decode(frame["data_base64"])
            image = Image.open(BytesIO(raw)).convert("RGB")
            image.thumbnail((560, 420), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.photo_refs[name] = photo
            self.image_labels[name].configure(image=photo, text="")

    @staticmethod
    def _format_values(values: list[Any]) -> str:
        return "[" + ", ".join(f"{float(value): .3f}" for value in values) + "]"

    def send_target(self) -> None:
        try:
            targets = [float(variable.get()) for variable in self.target_vars]
        except ValueError:
            messagebox.showerror("Invalid target", "Enter a number in all six fields.")
            return
        seq = self.client.send_target(targets)
        self.status_var.set(f"Sent command {seq}; waiting for simulator acknowledgement…")

    def hold_pose(self) -> None:
        if self.latest_observation is None:
            messagebox.showinfo("No state yet", "Wait for the first feedback frame.")
            return
        positions = self.latest_observation["state"]["joint_position_rad"]
        for variable, position in zip(self.target_vars, positions):
            variable.set(f"{float(position):.5f}")
        self.send_target()

    def close(self) -> None:
        self.client.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor and command a MuJoCo bridge.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8765",
        help="Bridge base URL; use https:// for an authenticated remote tunnel.",
    )
    parser.add_argument(
        "--token-env",
        default="MUJOCO_BRIDGE_TOKEN",
        help="Environment variable containing the bridge token, if configured.",
    )
    args = parser.parse_args()

    try:
        client = BridgeClient(args.base_url, os.environ.get(args.token_env) or None)
    except ValueError as error:
        parser.error(str(error))
    BridgeMonitor(client).run()


if __name__ == "__main__":
    main()
