from __future__ import annotations

import json
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
from PIL import Image, ImageTk

from offline_waybill_test import (
    build_batch_report,
    find_images,
    result_payload,
    write_csv,
)
from project0714_grasp.waybill_inspection import AsyncWaybillInspector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(r"G:\CQ\datasets\shipping_label_2.0_8.31\test")
IMAGE_SIZE = (520, 350)


class WaybillDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("快递面单条形码识别系统")
        self.root.geometry("1440x900")
        self.root.minsize(1180, 760)

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.rows: dict[str, dict[str, Any]] = {}
        self.preview_images: list[ImageTk.PhotoImage] = []
        self.success_count = 0
        self.processed_count = 0
        self.total_count = 0
        self.elapsed_total = 0.0

        self.input_var = tk.StringVar(value=str(DEFAULT_INPUT))
        self.output_var = tk.StringVar(
            value=str(
                PROJECT_ROOT
                / "calibration_suite"
                / "workspace"
                / "waybill_desktop"
            )
        )
        self.waybill_model_var = tk.StringVar(
            value=str(PROJECT_ROOT / "weights" / "best.pt")
        )
        self.barcode_model_var = tk.StringVar(
            value=str(PROJECT_ROOT / "weights" / "barcode_weights" / "best.pt")
        )
        self.waybill_conf_var = tk.StringVar(value="0.50")
        self.barcode_conf_var = tk.StringVar(value="0.25")
        self.status_var = tk.StringVar(value="就绪")
        self.total_var = tk.StringVar(value="总数 0")
        self.success_var = tk.StringVar(value="成功 0")
        self.failed_var = tk.StringVar(value="失败 0")
        self.rate_var = tk.StringVar(value="成功率 0.00%")
        self.average_var = tk.StringVar(value="平均耗时 0.000s")
        self.current_result_var = tk.StringVar(value="等待识别")

        self._configure_style()
        self._build_ui()
        self.root.after(100, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Metric.TLabel", font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Result.TLabel", font=("Consolas", 16, "bold"))
        style.configure("TButton", font=("Microsoft YaHei UI", 10))
        style.configure("Treeview", rowheight=26, font=("Microsoft YaHei UI", 9))
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="快递面单条形码识别系统", style="Title.TLabel").pack(
            side="left"
        )
        ttk.Label(header, textvariable=self.status_var).pack(side="right", padx=8)

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)
        run_page = ttk.Frame(notebook, padding=10)
        settings_page = ttk.Frame(notebook, padding=14)
        notebook.add(run_page, text="识别工作台")
        notebook.add(settings_page, text="模型与参数")
        self._build_run_page(run_page)
        self._build_settings_page(settings_page)

    def _build_run_page(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="任务", padding=8)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="图片/目录").grid(row=0, column=0, padx=5, sticky="w")
        ttk.Entry(controls, textvariable=self.input_var).grid(
            row=0, column=1, padx=5, sticky="ew"
        )
        ttk.Button(controls, text="选择图片", command=self._choose_file).grid(
            row=0, column=2, padx=3
        )
        ttk.Button(controls, text="选择目录", command=self._choose_directory).grid(
            row=0, column=3, padx=3
        )
        self.start_button = ttk.Button(controls, text="开始识别", command=self._start)
        self.start_button.grid(row=0, column=4, padx=(12, 3))
        self.stop_button = ttk.Button(
            controls, text="停止", command=self._stop, state="disabled"
        )
        self.stop_button.grid(row=0, column=5, padx=3)
        ttk.Button(controls, text="打开结果目录", command=self._open_output).grid(
            row=0, column=6, padx=3
        )

        metric_frame = ttk.Frame(parent)
        metric_frame.pack(fill="x", pady=10)
        for text_var in (
            self.total_var,
            self.success_var,
            self.failed_var,
            self.rate_var,
            self.average_var,
        ):
            ttk.Label(metric_frame, textvariable=text_var, style="Metric.TLabel").pack(
                side="left", padx=(4, 28)
            )
        self.progress = ttk.Progressbar(metric_frame, mode="determinate")
        self.progress.pack(side="right", fill="x", expand=True, padx=8)

        preview_frame = ttk.Frame(parent)
        preview_frame.pack(fill="x")
        original_box = ttk.LabelFrame(preview_frame, text="原始图片", padding=5)
        original_box.pack(side="left", fill="both", expand=True, padx=(0, 5))
        roi_box = ttk.LabelFrame(preview_frame, text="YOLO面单 ROI", padding=5)
        roi_box.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.original_label = ttk.Label(original_box, anchor="center")
        self.original_label.pack(fill="both", expand=True)
        self.roi_label = ttk.Label(roi_box, anchor="center")
        self.roi_label.pack(fill="both", expand=True)

        result_bar = ttk.Frame(parent)
        result_bar.pack(fill="x", pady=(8, 4))
        ttk.Label(result_bar, text="当前结果：").pack(side="left")
        ttk.Label(
            result_bar, textvariable=self.current_result_var, style="Result.TLabel"
        ).pack(side="left", padx=8)

        table_box = ttk.LabelFrame(parent, text="逐张识别结果", padding=5)
        table_box.pack(fill="both", expand=True)
        columns = ("filename", "status", "barcode", "elapsed", "waybill", "error")
        self.tree = ttk.Treeview(table_box, columns=columns, show="headings")
        headings = {
            "filename": "文件名",
            "status": "状态",
            "barcode": "Code128",
            "elapsed": "耗时(s)",
            "waybill": "面单帧",
            "error": "说明",
        }
        widths = {
            "filename": 150,
            "status": 80,
            "barcode": 190,
            "elapsed": 85,
            "waybill": 75,
            "error": 500,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="w")
        scrollbar = ttk.Scrollbar(table_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.tag_configure("success", foreground="#087f23")
        self.tree.tag_configure("failed", foreground="#b71c1c")
        self.tree.bind("<<TreeviewSelect>>", self._select_result)

    def _build_settings_page(self, parent: ttk.Frame) -> None:
        form = ttk.LabelFrame(parent, text="模型和识别参数", padding=14)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        fields = (
            ("面单 Detect 模型", self.waybill_model_var, "file"),
            ("条码 OBB 模型", self.barcode_model_var, "file"),
            ("结果目录", self.output_var, "directory"),
            ("面单置信度", self.waybill_conf_var, None),
            ("条码 OBB 置信度", self.barcode_conf_var, None),
        )
        for row, (label, variable, browse_kind) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=row, column=0, padx=6, pady=7, sticky="w")
            ttk.Entry(form, textvariable=variable).grid(
                row=row, column=1, padx=6, pady=7, sticky="ew"
            )
            if browse_kind == "file":
                command = lambda var=variable: self._choose_model(var)
                ttk.Button(form, text="浏览", command=command).grid(
                    row=row, column=2, padx=6
                )
            elif browse_kind == "directory":
                ttk.Button(form, text="浏览", command=self._choose_output).grid(
                    row=row, column=2, padx=6
                )

        note = (
            "识别链路：面单 Detect → 面单 ROI → 条码 OBB → 非对称外扩/透视展开 → "
            "ZXing Code128 → pure/Binarizer → 传统定位兜底。\n"
            "设置在下一次任务启动时生效；密码和机器人控制不在此离线识别界面中处理。"
        )
        ttk.Label(parent, text=note, wraplength=1050, justify="left").pack(
            fill="x", pady=18
        )

    def _choose_file(self) -> None:
        value = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if value:
            self.input_var.set(value)

    def _choose_directory(self) -> None:
        value = filedialog.askdirectory(title="选择图片目录")
        if value:
            self.input_var.set(value)

    def _choose_output(self) -> None:
        value = filedialog.askdirectory(title="选择结果目录")
        if value:
            self.output_var.set(value)

    def _choose_model(self, variable: tk.StringVar) -> None:
        value = filedialog.askopenfilename(
            title="选择模型", filetypes=[("PyTorch weights", "*.pt")]
        )
        if value:
            variable.set(value)

    def _validated_settings(self) -> dict[str, Any] | None:
        input_path = Path(self.input_var.get().strip()).expanduser().resolve()
        waybill_model = Path(self.waybill_model_var.get().strip()).expanduser().resolve()
        barcode_model = Path(self.barcode_model_var.get().strip()).expanduser().resolve()
        if not input_path.exists():
            messagebox.showerror("输入错误", f"图片或目录不存在：\n{input_path}")
            return None
        for label, path in (("面单模型", waybill_model), ("条码模型", barcode_model)):
            if not path.is_file():
                messagebox.showerror("模型错误", f"{label}不存在：\n{path}")
                return None
        try:
            waybill_conf = float(self.waybill_conf_var.get())
            barcode_conf = float(self.barcode_conf_var.get())
            if not 0.0 < waybill_conf <= 1.0 or not 0.0 < barcode_conf <= 1.0:
                raise ValueError
        except ValueError:
            messagebox.showerror("参数错误", "置信度必须是0到1之间的数字。")
            return None
        images = find_images(input_path, recursive=True)
        if not images:
            messagebox.showerror("输入错误", "没有找到支持的图片。")
            return None
        return {
            "images": images,
            "output": Path(self.output_var.get().strip()).expanduser().resolve(),
            "waybill_model": waybill_model,
            "barcode_model": barcode_model,
            "waybill_conf": waybill_conf,
            "barcode_conf": barcode_conf,
        }

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        settings = self._validated_settings()
        if settings is None:
            return
        self.stop_event.clear()
        self.rows.clear()
        self.tree.delete(*self.tree.get_children())
        self.success_count = 0
        self.processed_count = 0
        self.elapsed_total = 0.0
        self.total_count = len(settings["images"])
        self.progress.configure(maximum=self.total_count, value=0)
        self._refresh_metrics()
        self.status_var.set("正在加载模型…")
        self.current_result_var.set("等待第一张结果")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.worker = threading.Thread(
            target=self._run_recognition, args=(settings,), daemon=True
        )
        self.worker.start()

    def _run_recognition(self, settings: dict[str, Any]) -> None:
        output: Path = settings["output"]
        output.mkdir(parents=True, exist_ok=True)
        summaries: list[dict[str, object]] = []
        try:
            inspector = AsyncWaybillInspector(
                camera_ip="offline",
                username="offline",
                password="offline",
                model_path=settings["waybill_model"],
                barcode_model_path=settings["barcode_model"],
                output_dir=output,
                confidence=settings["waybill_conf"],
                barcode_confidence=settings["barcode_conf"],
                allow_single_frame_result=True,
            )
        except Exception as exc:
            self.events.put(("fatal", f"模型加载失败：{exc}"))
            return

        self.events.put(("status", "模型已加载，正在识别…"))
        try:
            for index, path in enumerate(settings["images"], start=1):
                if self.stop_event.is_set():
                    break
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    payload = {
                        "label": path.name,
                        "images": [str(path)],
                        "has_waybill": False,
                        "barcode": None,
                        "frame_count": 0,
                        "waybill_frame_count": 0,
                        "elapsed_s": 0.0,
                        "detection_to_barcode_s": None,
                        "error": "OpenCV无法读取图片",
                    }
                    summaries.append(payload)
                    self.events.put(("result", (payload, path, None)))
                    continue
                result = inspector._inspect_frames(index, [image])
                payload = result_payload(path.name, [path], result)
                summaries.append(payload)
                run_dirs = list(output.glob(f"candidate_{index}_*"))
                run_dir = max(run_dirs, key=lambda item: item.stat().st_mtime) if run_dirs else None
                roi_path = None
                if run_dir is not None:
                    roi_files = sorted(run_dir.glob("waybill_*_*.jpg"))
                    roi_path = roi_files[0] if roi_files else None
                self.events.put(("result", (payload, path, roi_path)))
        finally:
            inspector.close()

        report = build_batch_report(summaries)
        (output / "summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "batch_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_csv(output / "results.csv", summaries)
        self.events.put(("done", (report, self.stop_event.is_set())))

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "status":
                    self.status_var.set(str(payload))
                elif event == "result":
                    self._add_result(*payload)
                elif event == "fatal":
                    self.status_var.set("启动失败")
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    messagebox.showerror("识别失败", str(payload))
                elif event == "done":
                    report, stopped = payload
                    self.status_var.set("已停止" if stopped else "任务完成")
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    if not stopped:
                        messagebox.showinfo(
                            "任务完成",
                            f"共处理 {report['total_images']} 张\n"
                            f"成功 {report['successful_images']} 张\n"
                            f"成功率 {report['success_rate_percent']:.2f}%\n"
                            f"平均耗时 {report['average_processing_time_s_all_images']} 秒",
                        )
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _add_result(
        self, payload: dict[str, Any], source_path: Path, roi_path: Path | None
    ) -> None:
        self.processed_count += 1
        success = bool(payload.get("barcode"))
        if success:
            self.success_count += 1
        elapsed = float(payload.get("elapsed_s") or 0.0)
        self.elapsed_total += elapsed
        item_id = self.tree.insert(
            "",
            "end",
            values=(
                payload.get("label"),
                "成功" if success else "失败",
                payload.get("barcode") or "",
                f"{elapsed:.3f}",
                payload.get("waybill_frame_count", 0),
                payload.get("error") or "",
            ),
            tags=("success" if success else "failed",),
        )
        self.rows[item_id] = {
            "source": source_path,
            "roi": roi_path,
            "payload": payload,
        }
        self.tree.see(item_id)
        self.tree.selection_set(item_id)
        self._show_images(source_path, roi_path)
        self.current_result_var.set(
            payload.get("barcode") or f"未识别：{payload.get('label')}"
        )
        self.progress.configure(value=self.processed_count)
        self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        failed = self.processed_count - self.success_count
        rate = self.success_count * 100.0 / self.processed_count if self.processed_count else 0.0
        average = self.elapsed_total / self.processed_count if self.processed_count else 0.0
        self.total_var.set(f"总数 {self.total_count}")
        self.success_var.set(f"成功 {self.success_count}")
        self.failed_var.set(f"失败 {failed}")
        self.rate_var.set(f"成功率 {rate:.2f}%")
        self.average_var.set(f"平均耗时 {average:.3f}s")

    def _select_result(self, _event: tk.Event[Any]) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        row = self.rows.get(selection[0])
        if row:
            self._show_images(row["source"], row["roi"])
            payload = row["payload"]
            self.current_result_var.set(
                payload.get("barcode") or f"未识别：{payload.get('label')}"
            )

    def _show_images(self, source: Path, roi: Path | None) -> None:
        self.preview_images.clear()
        self._set_preview(self.original_label, source, "原图不可用")
        self._set_preview(self.roi_label, roi, "没有面单ROI")

    def _set_preview(
        self, label: ttk.Label, path: Path | None, empty_text: str
    ) -> None:
        if path is None or not path.is_file():
            label.configure(image="", text=empty_text)
            return
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.preview_images.append(photo)
            label.configure(image=photo, text="")
        except Exception:
            label.configure(image="", text=empty_text)

    def _stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("正在停止，将在当前图片完成后结束…")
        self.stop_button.configure(state="disabled")

    def _open_output(self) -> None:
        path = Path(self.output_var.get().strip()).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)  # type: ignore[attr-defined]

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno("退出", "识别任务仍在运行，确定退出吗？"):
                return
            self.stop_event.set()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    WaybillDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
