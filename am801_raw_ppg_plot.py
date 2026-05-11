import argparse
import csv
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

import am801_protocol as proto


@dataclass
class RawSample:
    wall_time: float
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
    wall_time = time.time()
    elapsed = wall_time - start_wall_time
    return RawSample(
        wall_time=wall_time,
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
                        "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(sample.wall_time))
                        + f".{int((sample.wall_time % 1) * 1000):03d}",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Live AM801 raw PPG plotter")
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument("--window-seconds", type=float, default=20.0, help="Visible plot window in seconds")
    parser.add_argument("--csv", type=Path, help="Optional CSV file to append timestamped samples to")
    parser.add_argument("--start", dest="start_stream", action="store_true", default=True, help="Send the start control command before plotting")
    parser.add_argument("--no-start", dest="start_stream", action="store_false", help="Do not send the start control command")
    parser.add_argument("--stop-on-exit", dest="stop_on_exit", action="store_true", default=True, help="Send the stop control command when the plot closes")
    parser.add_argument("--no-stop-on-exit", dest="stop_on_exit", action="store_false", help="Leave the device streaming when the plot closes")
    args = parser.parse_args()

    samples: deque[RawSample] = deque(maxlen=5000)
    lock = threading.Lock()
    stop_event = threading.Event()
    csv_file = None
    csv_writer = None
    start_wall_time = time.time()

    with proto.open_serial(args.port, args.baud) as ser:
        print(f"Opened {args.port} @ {args.baud} baud")

        if args.start_stream:
            sent = proto.send_command(ser, proto.CMD_CONTROL, b"\x01")
            print(f"sent: {sent.hex(' ')}")

        if args.csv is not None:
            csv_file = args.csv.open("w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=["wall_time", "elapsed_s", "red", "infrared", "background"],
            )
            csv_writer.writeheader()

        worker = threading.Thread(
            target=capture_raw_samples,
            args=(ser, stop_event, start_wall_time, samples, lock, csv_writer),
            daemon=True,
        )
        worker.start()

        figure, axes, lines = build_plot()
        figure.suptitle(f"AM801 raw stream on {args.port} @ {args.baud} baud")

        try:
            while plt.fignum_exists(figure.number):
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())