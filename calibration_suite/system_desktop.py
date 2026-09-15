from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PROJECT_ROOT / "calibration_suite" / "surface_cluster_grasp.py"
OFFLINE_GUI = PROJECT_ROOT / "calibration_suite" / "waybill_desktop.py"
DEFAULT_DEBUG_IMAGE = (
    PROJECT_ROOT / "calibration_suite" / "workspace" / "surface_grasp" / "debug_masks.png"
)
DEFAULT_WAYBILL_OUTPUT = (
    PROJECT_ROOT / "calibration_suite" / "workspace" / "waybill_inspection"
)
DEFAULT_GUI_FRAME_DIR = PROJECT_ROOT / "calibration_suite" / "workspace" / "system_gui_frames"


class SystemDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("智能包裹抓取与条码识别系统")
        self.root.geometry("1520x920")
        self.root.minsize(1240, 760)

        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.log_lines: list[str] = []
        self.color_photo: ImageTk.PhotoImage | None = None
        self.depth_photo: ImageTk.PhotoImage | None = None
        self.frame_mtimes = {"color.jpg": 0.0, "depth.jpg": 0.0}
        self.last_waybill_mtime = 0.0

        self.mode_var = tk.StringVar(value="安全演练")
        self.system_var = tk.StringVar(value="未启动")
        self.camera_var = tk.StringVar(value="未知")
        self.robot_var = tk.StringVar(value="未连接")
        self.analysis_var = tk.StringVar(value="等待")
        self.motion_var = tk.StringVar(value="停止")
        self.package_var = tk.StringVar(value="0")
        self.barcode_var = tk.StringVar(value="--")
        self.last_event_var = tk.StringVar(value="等待启动")

        self.robot_ip_var = tk.StringVar(value="192.168.2.160")
        self.hik_ip_var = tk.StringVar(value="192.168.2.14")
        self.hik_user_var = tk.StringVar(value="admin")
        self.hik_password_var = tk.StringVar(value=os.environ.get("HIKVISION_PASSWORD", ""))
        self.package_model_var = tk.StringVar(
            value=str(
                PROJECT_ROOT
                / "project_yolo_train_0804"
                / "runs"
                / "yolo11s_obb_2.0"
                / "weights"
                / "best.pt"
            )
        )
        self.waybill_model_var = tk.StringVar(value=str(PROJECT_ROOT / "weights" / "best.pt"))
        self.barcode_model_var = tk.StringVar(
            value=str(PROJECT_ROOT / "weights" / "barcode_weights" / "best.pt")
        )
        self.yolo_conf_var = tk.StringVar(value="0.35")
        self.waybill_conf_var = tk.StringVar(value="0.50")
        self.barcode_conf_var = tk.StringVar(value="0.25")
        self.speed_var = tk.StringVar(value="400")
        self.standoff_var = tk.StringVar(value="100")
        self.pickup_down_var = tk.StringVar(value="110")
        self.primary_port_var = tk.StringVar(value="6")
        self.secondary_port_var = tk.StringVar(value="5")
        self.enable_waybill_var = tk.BooleanVar(value=True)
        self.dry_run_var = tk.BooleanVar(value=True)
        self.ack_tcp_var = tk.BooleanVar(value=False)
        self.extra_args_var = tk.StringVar(value="")

        self._configure_style()
        self._build_ui()
        self.root.after(100, self._poll_events)
        self.root.after(1000, self._refresh_artifacts)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 19, "bold"))
        style.configure("CardTitle.TLabel", font=("Microsoft YaHei UI", 9))
        style.configure("CardValue.TLabel", font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("Barcode.TLabel", font=("Consolas", 20, "bold"))
        style.configure("Danger.TButton", font=("Microsoft YaHei UI", 12, "bold"))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="智能包裹抓取与条码识别系统", style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.mode_var).pack(side="right", padx=12)

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)
        dashboard = ttk.Frame(notebook, padding=10)
        settings = ttk.Frame(notebook, padding=12)
        logs = ttk.Frame(notebook, padding=8)
        notebook.add(dashboard, text="运行控制台")
        notebook.add(settings, text="设备与参数")
        notebook.add(logs, text="完整日志")
        self._build_dashboard(dashboard)
        self._build_settings(settings)
        self._build_logs(logs)

    def _build_dashboard(self, parent: ttk.Frame) -> None:
        cards = ttk.Frame(parent)
        cards.pack(fill="x")
        for title, variable in (
            ("系统", self.system_var),
            ("顶部相机", self.camera_var),
            ("机械臂", self.robot_var),
            ("视觉分析", self.analysis_var),
            ("运动状态", self.motion_var),
            ("候选包裹", self.package_var),
        ):
            card = ttk.LabelFrame(cards, padding=10)
            card.pack(side="left", fill="x", expand=True, padx=4)
            ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=variable, style="CardValue.TLabel").pack(anchor="w", pady=(5, 0))

        controls = ttk.LabelFrame(parent, text="整机控制", padding=10)
        controls.pack(fill="x", pady=10)
        self.launch_button = ttk.Button(controls, text="启动系统", command=self._launch_system)
        self.launch_button.pack(side="left", padx=4)
        self.analyze_button = ttk.Button(controls, text="重新分析", command=lambda: self._send("ANALYZE"), state="disabled")
        self.analyze_button.pack(side="left", padx=4)
        self.start_button = ttk.Button(controls, text="开始连续抓取", command=lambda: self._send("START"), state="disabled")
        self.start_button.pack(side="left", padx=4)
        self.reset_button = ttk.Button(controls, text="刷新场景", command=lambda: self._send("RESET"), state="disabled")
        self.reset_button.pack(side="left", padx=4)
        self.stop_button = ttk.Button(controls, text="软件停止", command=lambda: self._send("STOP"), state="disabled", style="Danger.TButton")
        self.stop_button.pack(side="left", padx=(24, 4))
        self.quit_button = ttk.Button(controls, text="关闭主程序", command=lambda: self._send("QUIT"), state="disabled")
        self.quit_button.pack(side="left", padx=4)
        ttk.Button(controls, text="离线条码测试", command=self._open_offline_gui).pack(side="right", padx=4)
        ttk.Button(controls, text="打开工作目录", command=self._open_workspace).pack(side="right", padx=4)

        content = ttk.Panedwindow(parent, orient="horizontal")
        content.pack(fill="both", expand=True)
        visual = ttk.Frame(content)
        activity = ttk.Frame(content)
        content.add(visual, weight=3)
        content.add(activity, weight=2)

        preview_row = ttk.Frame(visual)
        preview_row.pack(fill="both", expand=True)
        color_box = ttk.LabelFrame(preview_row, text="顶部RGB / 包裹检测", padding=5)
        depth_box = ttk.LabelFrame(preview_row, text="深度 / 抓取候选", padding=5)
        color_box.pack(side="left", fill="both", expand=True, padx=(0, 4))
        depth_box.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.color_label = ttk.Label(color_box, text="等待相机画面", anchor="center")
        self.depth_label = ttk.Label(depth_box, text="等待深度画面", anchor="center")
        self.color_label.pack(fill="both", expand=True)
        self.depth_label.pack(fill="both", expand=True)

        barcode_box = ttk.LabelFrame(visual, text="最新物流条码", padding=10)
        barcode_box.pack(fill="x", pady=(8, 0))
        ttk.Label(barcode_box, textvariable=self.barcode_var, style="Barcode.TLabel").pack(side="left")
        ttk.Label(barcode_box, textvariable=self.last_event_var).pack(side="right", padx=8)

        event_box = ttk.LabelFrame(activity, text="关键事件", padding=5)
        event_box.pack(fill="both", expand=True)
        self.event_tree = ttk.Treeview(event_box, columns=("time", "type", "detail"), show="headings")
        self.event_tree.heading("time", text="时间")
        self.event_tree.heading("type", text="类型")
        self.event_tree.heading("detail", text="内容")
        self.event_tree.column("time", width=80, anchor="w")
        self.event_tree.column("type", width=90, anchor="w")
        self.event_tree.column("detail", width=430, anchor="w")
        scroll = ttk.Scrollbar(event_box, orient="vertical", command=self.event_tree.yview)
        self.event_tree.configure(yscrollcommand=scroll.set)
        self.event_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        ttk.Label(
            activity,
            text="注意：软件停止不能替代机械臂物理急停。真实运动前必须完成TCP、固定路径和安全区域验证。",
            foreground="#b71c1c",
            wraplength=530,
        ).pack(fill="x", pady=8)

    def _build_settings(self, parent: ttk.Frame) -> None:
        safety = ttk.LabelFrame(parent, text="运行模式", padding=10)
        safety.pack(fill="x")
        ttk.Checkbutton(safety, text="安全演练（不向机械臂发送运动）", variable=self.dry_run_var, command=self._update_mode).pack(side="left", padx=6)
        ttk.Checkbutton(safety, text="启用底部面单与条码识别", variable=self.enable_waybill_var).pack(side="left", padx=20)
        ttk.Checkbutton(safety, text="已完成TCP及固定路径物理验证", variable=self.ack_tcp_var).pack(side="left", padx=20)

        columns = ttk.Frame(parent)
        columns.pack(fill="both", expand=True, pady=10)
        device = ttk.LabelFrame(columns, text="设备和模型", padding=10)
        motion = ttk.LabelFrame(columns, text="运动和识别参数", padding=10)
        device.pack(side="left", fill="both", expand=True, padx=(0, 5))
        motion.pack(side="left", fill="both", expand=True, padx=(5, 0))
        device.columnconfigure(1, weight=1)
        motion.columnconfigure(1, weight=1)

        self._entry(device, 0, "机器人IP", self.robot_ip_var)
        self._entry(device, 1, "海康相机IP", self.hik_ip_var)
        self._entry(device, 2, "海康用户名", self.hik_user_var)
        self._entry(device, 3, "海康密码", self.hik_password_var, show="●")
        self._path_entry(device, 4, "包裹YOLO模型", self.package_model_var)
        self._path_entry(device, 5, "面单Detect模型", self.waybill_model_var)
        self._path_entry(device, 6, "条码OBB模型", self.barcode_model_var)

        self._entry(motion, 0, "机械臂速度 mm/s", self.speed_var)
        self._entry(motion, 1, "安全悬停距离 mm", self.standoff_var)
        self._entry(motion, 2, "下降距离 mm", self.pickup_down_var)
        self._entry(motion, 3, "主吸盘DO端口", self.primary_port_var)
        self._entry(motion, 4, "副吸盘DO端口", self.secondary_port_var)
        self._entry(motion, 5, "包裹YOLO阈值", self.yolo_conf_var)
        self._entry(motion, 6, "面单阈值", self.waybill_conf_var)
        self._entry(motion, 7, "条码OBB阈值", self.barcode_conf_var)
        self._entry(motion, 8, "附加命令行参数", self.extra_args_var)

        ttk.Label(
            parent,
            text="参数只在下一次启动主程序时生效。海康密码通过环境变量传递，不显示在日志或命令预览中。",
        ).pack(anchor="w", pady=4)

    def _build_logs(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(toolbar, text="清空显示", command=self._clear_log).pack(side="left")
        ttk.Button(toolbar, text="保存日志", command=self._save_log).pack(side="left", padx=6)
        self.log_text = tk.Text(parent, wrap="none", font=("Consolas", 9), bg="#111827", fg="#e5e7eb", insertbackground="white")
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=self.log_text.yview)
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")

    @staticmethod
    def _entry(parent: ttk.Frame, row: int, label: str, variable: tk.Variable, show: str | None = None) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=6)
        ttk.Entry(parent, textvariable=variable, show=show).grid(row=row, column=1, sticky="ew", padx=5, pady=6)

    def _path_entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        self._entry(parent, row, label, variable)
        ttk.Button(parent, text="浏览", command=lambda: self._choose_model(variable)).grid(row=row, column=2, padx=4)

    def _choose_model(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch weights", "*.pt")])
        if path:
            variable.set(path)

    def _update_mode(self) -> None:
        self.mode_var.set("安全演练" if self.dry_run_var.get() else "真实机械臂")

    def _validated_command(self) -> tuple[list[str], dict[str, str]] | None:
        model_vars = (self.package_model_var, self.waybill_model_var, self.barcode_model_var)
        for variable in model_vars:
            path = Path(variable.get().strip()).expanduser().resolve()
            if not path.is_file():
                messagebox.showerror("模型不存在", str(path))
                return None
        try:
            speed = float(self.speed_var.get())
            standoff = float(self.standoff_var.get())
            pickup_down = float(self.pickup_down_var.get())
            yolo_conf = float(self.yolo_conf_var.get())
            waybill_conf = float(self.waybill_conf_var.get())
            barcode_conf = float(self.barcode_conf_var.get())
            primary_port = int(self.primary_port_var.get())
            secondary_port = int(self.secondary_port_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "速度、距离、端口和置信度必须是有效数字。")
            return None
        if speed <= 0 or standoff < 0 or pickup_down < 0:
            messagebox.showerror("参数错误", "速度必须大于0，距离不能小于0。")
            return None
        if not all(0 < value <= 1 for value in (yolo_conf, waybill_conf, barcode_conf)):
            messagebox.showerror("参数错误", "置信度必须在0到1之间。")
            return None
        if primary_port == secondary_port:
            messagebox.showerror("参数错误", "主、副吸盘不能使用同一个DO端口。")
            return None
        if not self.dry_run_var.get() and not self.ack_tcp_var.get():
            messagebox.showerror("安全锁定", "真实运动前必须勾选已完成TCP及固定路径物理验证。")
            return None
        if not self.dry_run_var.get():
            confirmed = messagebox.askyesno(
                "确认真实运动",
                f"即将连接机械臂 {self.robot_ip_var.get()} 并允许真实运动。\n"
                "请确认人员已离开工作空间，物理急停可用。是否继续？",
            )
            if not confirmed:
                return None

        cmd = [
            str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
            "-u",
            str(MAIN_SCRIPT),
            "--control-stdin",
            "--headless",
            "--gui-frame-dir", str(DEFAULT_GUI_FRAME_DIR),
            "--tool-name", "tool4",
            "--wobj-name", "wobj0",
            "--tcp-override-xyz-mm", "-125.370", "87.591", "279.781",
            "--tcp-override-rpy-deg", "178.740", "-32.430", "-87.920",
            "--rpy-mode", "align-normal",
            "--normal-mode", "top-plane",
            "--tool-contact-axis", "minus-z",
            "--robot-ip", self.robot_ip_var.get().strip(),
            "--robot-speed-mm-s", str(speed),
            "--standoff-mm", str(standoff),
            "--pickup-down-mm", str(pickup_down),
            "--suction-do-port", str(primary_port),
            "--secondary-suction-do-port", str(secondary_port),
            "--secondary-suction-offset-y-mm", "-245",
            "--disable-third-suction",
            "--disable-fourth-suction",
            "--cup-volume-check",
            "--yolo-model", str(Path(self.package_model_var.get()).resolve()),
            "--yolo-conf", str(yolo_conf),
            "--waybill-model", str(Path(self.waybill_model_var.get()).resolve()),
            "--waybill-conf", str(waybill_conf),
            "--barcode-model", str(Path(self.barcode_model_var.get()).resolve()),
            "--barcode-conf", str(barcode_conf),
        ]
        if self.dry_run_var.get():
            cmd.append("--dry-run")
        else:
            cmd.append("--acknowledge-verified-tcp")
        if self.enable_waybill_var.get():
            if not self.hik_password_var.get():
                messagebox.showerror("相机密码缺失", "启用面单识别时必须填写海康相机密码。")
                return None
            cmd.extend([
                "--enable-waybill-inspection",
                "--waybill-camera-ip", self.hik_ip_var.get().strip(),
                "--waybill-camera-username", self.hik_user_var.get().strip(),
            ])
        extra = self.extra_args_var.get().strip()
        if extra:
            import shlex
            cmd.extend(shlex.split(extra, posix=False))
        environment = os.environ.copy()
        if self.hik_password_var.get():
            environment["HIKVISION_PASSWORD"] = self.hik_password_var.get()
        environment["PYTHONUNBUFFERED"] = "1"
        return cmd, environment

    def _launch_system(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        validated = self._validated_command()
        if validated is None:
            return
        cmd, environment = validated
        self._clear_runtime_state()
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception as exc:
            messagebox.showerror("启动失败", str(exc))
            return
        self.system_var.set("启动中")
        self.motion_var.set("停止")
        self._set_controls(True)
        self.reader_thread = threading.Thread(target=self._read_process, daemon=True)
        self.reader_thread.start()
        self._event("系统", "主程序已启动，等待相机与模型初始化")

    def _read_process(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.events.put(("log", line.rstrip()))
        code = process.wait()
        self.events.put(("exit", code))

    def _send(self, command: str) -> None:
        if self.process is None or self.process.poll() is not None or self.process.stdin is None:
            messagebox.showwarning("系统未运行", "请先启动主程序。")
            return
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            self._event("控制", command)
            if command == "STOP":
                self.motion_var.set("停止请求")
            elif command == "START":
                self.motion_var.set("自动运行")
        except Exception as exc:
            messagebox.showerror("命令发送失败", str(exc))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._handle_log(str(payload))
                elif kind == "exit":
                    self.system_var.set(f"已退出({payload})")
                    self.motion_var.set("停止")
                    self._set_controls(False)
                    self._event("系统", f"主程序退出，代码 {payload}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _handle_log(self, line: str) -> None:
        self.log_lines.append(line)
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        if "YOLO loaded:" in line:
            self.system_var.set("视觉模型就绪")
        if "Robot connected:" in line:
            self.robot_var.set("已连接")
            self._event("设备", line)
        if "Robot connection/setup failed" in line:
            self.robot_var.set("连接失败")
            self._event("故障", line)
        if "Initial analysis failed" in line or "Analysis failed" in line:
            self.analysis_var.set("失败")
            self._event("视觉", line)
        match = re.search(r"Analysis (?:complete|refreshed #\d+): (\d+) candidate", line)
        if match:
            count = match.group(1)
            self.package_var.set(count)
            self.analysis_var.set("完成")
            self.camera_var.set("在线")
            self._event("视觉", f"检测到 {count} 个包裹候选")
        if "Starting batch candidate" in line or "Moving candidate" in line:
            self.motion_var.set("运动中")
            self._event("运动", line)
        if "Completed batch candidate" in line:
            self.motion_var.set("单件完成")
            self._event("完成", line)
        if "GUI STOP pressed" in line or "stopping robot motion" in line:
            self.motion_var.set("已停止")
            self._event("停止", line)
        barcode_match = re.search(r"barcode=([^\s,)]+)", line, re.IGNORECASE)
        if barcode_match:
            self.barcode_var.set(barcode_match.group(1))
            self._event("条码", barcode_match.group(1))
        if "ERROR" in line or "failed" in line.lower():
            self.last_event_var.set(line[-110:])

    def _event(self, event_type: str, detail: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self.event_tree.insert("", "end", values=(now, event_type, detail))
        children = self.event_tree.get_children()
        if len(children) > 500:
            self.event_tree.delete(children[0])
        self.event_tree.see(self.event_tree.get_children()[-1])
        self.last_event_var.set(detail[-100:])

    def _refresh_artifacts(self) -> None:
        try:
            for filename, label, attribute in (
                ("color.jpg", self.color_label, "color_photo"),
                ("depth.jpg", self.depth_label, "depth_photo"),
            ):
                path = DEFAULT_GUI_FRAME_DIR / filename
                if path.is_file() and path.stat().st_mtime > self.frame_mtimes[filename]:
                    self.frame_mtimes[filename] = path.stat().st_mtime
                    image = Image.open(path).convert("RGB")
                    image.thumbnail((560, 460), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    setattr(self, attribute, photo)
                    label.configure(image=photo, text="")
            results_path = DEFAULT_WAYBILL_OUTPUT / "results.jsonl"
            if results_path.is_file() and results_path.stat().st_mtime > self.last_waybill_mtime:
                self.last_waybill_mtime = results_path.stat().st_mtime
                lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if lines:
                    latest = json.loads(lines[-1])
                    if latest.get("barcode"):
                        self.barcode_var.set(str(latest["barcode"]))
        except Exception:
            pass
        self.root.after(1000, self._refresh_artifacts)

    def _set_controls(self, running: bool) -> None:
        self.launch_button.configure(state="disabled" if running else "normal")
        state = "normal" if running else "disabled"
        for button in (self.analyze_button, self.start_button, self.reset_button, self.stop_button, self.quit_button):
            button.configure(state=state)

    def _clear_runtime_state(self) -> None:
        self.system_var.set("启动中")
        self.camera_var.set("初始化")
        self.robot_var.set("演练模式" if self.dry_run_var.get() else "连接中")
        self.analysis_var.set("等待")
        self.motion_var.set("停止")
        self.package_var.set("0")
        self.barcode_var.set("--")

    def _open_offline_gui(self) -> None:
        subprocess.Popen([str(PROJECT_ROOT / ".venv" / "Scripts" / "pythonw.exe"), str(OFFLINE_GUI)], cwd=str(PROJECT_ROOT))

    @staticmethod
    def _open_workspace() -> None:
        workspace = PROJECT_ROOT / "calibration_suite" / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        os.startfile(workspace)  # type: ignore[attr-defined]

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", "end")

    def _save_log(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".log",
            initialfile=f"system_{datetime.now():%Y%m%d_%H%M%S}.log",
            filetypes=[("Log", "*.log"), ("Text", "*.txt")],
        )
        if path:
            Path(path).write_text("\n".join(self.log_lines), encoding="utf-8")

    def _on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("退出", "主程序仍在运行。是否先发送停止和退出命令？"):
                return
            self._send("STOP")
            self._send("QUIT")
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SystemDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
