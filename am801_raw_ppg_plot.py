import argparse
import csv
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from serial.tools import list_ports

import am801_protocol as proto


@dataclass
class RawSample:
    host_timestamp_ns: int
    host_timestamp_iso: str
    elapsed: float
    red: int
    infrared: int
    background: int


def extract_raw_sample(packet: proto.Packet, start_wall_time: float) -> RawSample | None:
    if not packet.valid_checksum or packet.command != proto.CMD_RAW or len(packet.payload) != 9:
        return None

    red = proto.u24_be(packet.payload[0:3])
    infrared = proto.u24_be(packet.payload[3:6])
    background = proto.u24_be(packet.payload[6:9])
    host_timestamp_ns = time.time_ns()
    host_timestamp_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
    elapsed = (host_timestamp_ns / 1_000_000_000) - start_wall_time
    return RawSample(
        host_timestamp_ns=host_timestamp_ns,
        host_timestamp_iso=host_timestamp_iso,
        elapsed=elapsed,
        red=red,
        infrared=infrared,
        background=background,
    )


def capture_raw_samples(
    ser,
    stop_event: threading.Event,
    start_wall_time: float,
    samples: deque[RawSample],
    lock: threading.Lock,
    csv_writer: csv.DictWriter | None,
) -> None:
    buffer = bytearray()
    while not stop_event.is_set():
        chunk = ser.read(256)
        if not chunk:
            continue

        buffer.extend(chunk)
        for packet in proto.parse_packets(buffer):
            sample = extract_raw_sample(packet, start_wall_time)
            if sample is None:
                continue

            with lock:
                samples.append(sample)

            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "host_timestamp_ns": sample.host_timestamp_ns,
                        "host_timestamp_iso": sample.host_timestamp_iso,
                        "elapsed_s": f"{sample.elapsed:.3f}",
                        "red": sample.red,
                        "infrared": sample.infrared,
                        "background": sample.background,
                    }
                )


def trim_window(values: list[float], window_seconds: float) -> list[float]:
    if window_seconds <= 0 or not values:
        return values

    cutoff = values[-1] - window_seconds
    start_index = 0
    for index, value in enumerate(values):
        if value >= cutoff:
            start_index = index
            break
    return values[start_index:]


def build_plot():
    plt.style.use("seaborn-v0_8-darkgrid")
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    lines = []
    colors = [("red", "#d32f2f"), ("infrared", "#f57c00"), ("background", "#455a64")]
    for axis, (label, color) in zip(axes, colors, strict=True):
        (line,) = axis.plot([], [], color=color, linewidth=1.2)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        lines.append(line)
    axes[-1].set_xlabel("Elapsed time (s)")
    figure.tight_layout()
    return figure, list(axes), lines


def detect_ch341_port() -> str:
    for port in list_ports.comports():
        description = (port.description or "").lower()
        manufacturer = (port.manufacturer or "").lower()
        hwid = (port.hwid or "").lower()
        if "usb-serial ch341" in description or "wch" in manufacturer or "ch341" in hwid:
            return port.device

    detected = [f"{port.device} ({port.description})" for port in list_ports.comports()]
    detected_text = ", ".join(detected) if detected else "none"
    raise RuntimeError(
        "Could not auto-detect a USB-SERIAL CH341 port. "
        f"Use --port explicitly. Detected ports: {detected_text}"
    )


def print_csv_preview(csv_path: Path, max_rows: int = 10) -> None:
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return

    tail: deque[dict[str, str]] = deque(maxlen=max_rows)
    row_count = 0
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            tail.append(row)

    print(f"CSV saved: {csv_path} ({row_count} rows)")
    if not tail:
        print("CSV has no sample rows.")
        return

    print("Last rows:")
    for row in tail:
        print(
            "  "
            f"{row.get('host_timestamp_iso', '')} "
            f"red={row.get('red', '')} "
            f"infrared={row.get('infrared', '')} "
            f"background={row.get('background', '')}"
        )


def print_samples_preview(samples: list[RawSample], max_rows: int = 10) -> None:
    print(f"Captured in memory: {len(samples)} rows")
    if not samples:
        print("No samples captured.")
        return

    print("Last rows:")
    for sample in samples[-max_rows:]:
        print(
            "  "
            f"{sample.host_timestamp_iso} "
            f"red={sample.red} "
            f"infrared={sample.infrared} "
            f"background={sample.background}"
        )


def default_csv_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"am801_raw_ppg_{timestamp}.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live AM801 raw PPG plotter")
    parser.add_argument("--port", help="Serial port (for example COM6). If omitted, auto-detect USB-SERIAL CH341")
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--window-seconds", type=float, default=20.0, help="Visible plot window in seconds")
    parser.add_argument("--csv", type=Path, help="CSV output path. If omitted, a timestamped CSV is created in the current folder")
    parser.add_argument("--start", dest="start_stream", action="store_true", default=True, help="Send the start control command before plotting")
    parser.add_argument("--no-start", dest="start_stream", action="store_false", help="Do not send the start control command")
    parser.add_argument("--stop-on-exit", dest="stop_on_exit", action="store_true", default=True, help="Send the stop control command when the plot closes")
    parser.add_argument("--no-stop-on-exit", dest="stop_on_exit", action="store_false", help="Leave the device streaming when the plot closes")
    parser.add_argument("--print-csv-on-exit", dest="print_csv_on_exit", action="store_true", default=True, help="Print captured CSV tail when stopping")
    parser.add_argument("--no-print-csv-on-exit", dest="print_csv_on_exit", action="store_false", help="Do not print CSV tail on exit")
    args = parser.parse_args()

    samples: deque[RawSample] = deque(maxlen=5000)
    lock = threading.Lock()
    stop_event = threading.Event()
    csv_file = None
    csv_writer = None
    start_wall_time = time.time()
    port = args.port or detect_ch341_port()
    csv_path = args.csv or default_csv_path()

    def request_stop(_signum=None, _frame=None):
        stop_event.set()
        plt.close("all")

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    with proto.open_serial(port, args.baud) as ser:
        print(f"Opened {port} @ {args.baud} baud")

        if args.start_stream:
            sent = proto.send_command(ser, proto.CMD_CONTROL, b"\x01")
            print(f"sent: {sent.hex(' ')}")

        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["host_timestamp_ns", "host_timestamp_iso", "elapsed_s", "red", "infrared", "background"],
        )
        csv_writer.writeheader()
        print(f"Saving CSV to: {csv_path.resolve()}")

        worker = threading.Thread(
            target=capture_raw_samples,
            args=(ser, stop_event, start_wall_time, samples, lock, csv_writer),
            daemon=True,
        )
        worker.start()

        figure, axes, lines = build_plot()
        figure.suptitle(f"AM801 raw stream on {port} @ {args.baud} baud")
        figure_number = int(getattr(figure, "number"))

        try:
            while plt.fignum_exists(figure_number) and not stop_event.is_set():
                with lock:
                    snapshot = list(samples)

                if snapshot:
                    times = [sample.elapsed for sample in snapshot]
                    red_values = [sample.red for sample in snapshot]
                    infrared_values = [sample.infrared for sample in snapshot]
                    background_values = [sample.background for sample in snapshot]

                    times = trim_window(times, args.window_seconds)
                    if args.window_seconds > 0:
                        cutoff = times[0]
                        start_index = 0
                        for index, value in enumerate([sample.elapsed for sample in snapshot]):
                            if value >= cutoff:
                                start_index = index
                                break
                        red_values = red_values[start_index:]
                        infrared_values = infrared_values[start_index:]
                        background_values = background_values[start_index:]

                    datasets = [red_values, infrared_values, background_values]
                    for axis, line, values in zip(axes, lines, datasets, strict=True):
                        line.set_data(times, values)
                        if times:
                            axis.set_xlim(times[0], max(times[-1], times[0] + 0.1))
                            lower = min(values)
                            upper = max(values)
                            if lower == upper:
                                axis.set_ylim(lower - 1, upper + 1)
                            else:
                                padding = max(1, int((upper - lower) * 0.05))
                                axis.set_ylim(lower - padding, upper + padding)

                figure.canvas.draw_idle()
                plt.pause(0.05)
        except KeyboardInterrupt:
            request_stop()
            print("Stopped.")
        finally:
            stop_event.set()
            worker.join(timeout=2.0)
            if args.stop_on_exit:
                try:
                    sent = proto.send_command(ser, proto.CMD_CONTROL, b"\x00")
                    print(f"sent: {sent.hex(' ')}")
                except Exception:
                    pass
            if csv_file is not None:
                csv_file.close()
            if args.print_csv_on_exit:
                print_csv_preview(csv_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())