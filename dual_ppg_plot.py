"""
Dual PPG Plotter: Simultaneous recording from Polar Verity Sense (BLE) and AM801 (serial).
Shows real-time plots for both devices. On Ctrl+C, saves each device to its own CSV.
"""

import asyncio
import argparse
import csv
import platform
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from bleak import BleakClient, BleakScanner
from serial.tools import list_ports

import am801_protocol as proto

# ─────────────────────────────────────────────────────────────────────
#  Verity Sense BLE constants & helpers  (from  verity_sense_ppg_raw.py)
# ─────────────────────────────────────────────────────────────────────

PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CP = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

GET_MEASUREMENT_SETTINGS = 0x01
REQUEST_MEASUREMENT_START = 0x02
STOP_MEASUREMENT = 0x03
GET_SDK_MODE_MEASUREMENT_SETTINGS = 0x04

PPG_MEASUREMENT_TYPE = 0x01
SDK_MODE_MEASUREMENT_TYPE = 0x09

SETTING_SAMPLE_RATE = 0x00
SETTING_RESOLUTION = 0x01
SETTING_RANGE = 0x02
SETTING_RANGE_MILLIUNIT = 0x03
SETTING_CHANNELS = 0x04
SETTING_FACTOR = 0x05
SETTING_SECURITY = 0x06

SETTING_FIELD_SIZE = {
    SETTING_SAMPLE_RATE: 2,
    SETTING_RESOLUTION: 2,
    SETTING_RANGE: 2,
    SETTING_RANGE_MILLIUNIT: 4,
    SETTING_CHANNELS: 1,
    SETTING_FACTOR: 4,
    SETTING_SECURITY: 16,
}

CP_STATUS_NAMES = {
    0: "SUCCESS",
    1: "ERROR_INVALID_OP_CODE",
    2: "ERROR_INVALID_MEASUREMENT_TYPE",
    3: "ERROR_NOT_SUPPORTED",
    4: "ERROR_INVALID_LENGTH",
    5: "ERROR_INVALID_PARAMETER",
    6: "ERROR_ALREADY_IN_STATE",
    7: "ERROR_INVALID_RESOLUTION",
    8: "ERROR_INVALID_SAMPLE_RATE",
    9: "ERROR_INVALID_RANGE",
    10: "ERROR_INVALID_MTU",
    11: "ERROR_INVALID_NUMBER_OF_CHANNELS",
    12: "ERROR_INVALID_STATE",
    13: "ERROR_DEVICE_IN_CHARGER",
    14: "ERROR_DISK_FULL",
}


class PmdController:
    def __init__(self):
        self.cp_queue = asyncio.Queue()

    def cp_notification_handler(self, _sender, data: bytearray):
        self.cp_queue.put_nowait(bytes(data))

    async def send_cp_command(self, client: BleakClient, opcode: int, params: bytes = b"") -> tuple[int, bytes]:
        packet = bytes([opcode]) + params
        await client.write_gatt_char(PMD_CP, packet, response=True)
        response, status, params = await self._wait_and_collect_cp_response(opcode)
        return status, params

    async def _wait_and_collect_cp_response(self, expected_opcode: int) -> tuple[bytes, int, bytes]:
        first = await self._wait_matching_cp_packet(expected_opcode)
        if len(first) < 4:
            raise RuntimeError(f"Short CP response: {first.hex(' ')}")
        status = first[3]
        if status != 0:
            return first, status, b""
        more = len(first) > 4 and first[4] != 0
        parameters = first[5:] if len(first) > 5 else b""
        while more:
            nxt = await self._wait_matching_cp_packet(expected_opcode)
            if len(nxt) > 5:
                parameters += nxt[5:]
            more = len(nxt) > 4 and nxt[4] != 0
        return first, status, parameters

    async def _wait_matching_cp_packet(self, expected_opcode: int) -> bytes:
        while True:
            pkt = await asyncio.wait_for(self.cp_queue.get(), timeout=20.0)
            if len(pkt) < 2:
                continue
            if pkt[0] != 0xF0:
                continue
            if pkt[1] != expected_opcode:
                continue
            return pkt


def parse_settings_blob(blob: bytes) -> dict[int, list[int]]:
    settings: dict[int, list[int]] = {}
    i = 0
    while i < len(blob):
        setting_type = blob[i]
        i += 1
        count = blob[i]
        i += 1
        field_size = SETTING_FIELD_SIZE.get(setting_type)
        if field_size is None:
            raise RuntimeError(f"Unknown setting type {setting_type}")
        vals = []
        for _ in range(count):
            chunk = blob[i: i + field_size]
            i += field_size
            if field_size == 1:
                vals.append(chunk[0])
            elif field_size in (2, 4):
                vals.append(int.from_bytes(chunk, byteorder="little", signed=False))
            elif field_size == 16:
                vals.append(int.from_bytes(chunk, byteorder="little", signed=False))
            else:
                vals.append(int.from_bytes(chunk, byteorder="little", signed=False))
        settings[setting_type] = vals
    return settings


def _append_setting_tlv(out: bytearray, setting_type: int, value: int) -> None:
    field_size = SETTING_FIELD_SIZE[setting_type]
    out += bytes([setting_type, 1])
    out += value.to_bytes(field_size, byteorder="little", signed=False)


def build_selected_settings_tlv(settings: dict[int, list[int]], sample_rate: int, resolution: int) -> bytes:
    out = bytearray()
    if SETTING_SAMPLE_RATE in settings:
        _append_setting_tlv(out, SETTING_SAMPLE_RATE, sample_rate)
    if SETTING_RESOLUTION in settings:
        _append_setting_tlv(out, SETTING_RESOLUTION, resolution)
    if SETTING_CHANNELS in settings and settings[SETTING_CHANNELS]:
        _append_setting_tlv(out, SETTING_CHANNELS, max(settings[SETTING_CHANNELS]))
    return bytes(out)


def to_signed24_le(b0: int, b1: int, b2: int) -> int:
    value = b0 | (b1 << 8) | (b2 << 16)
    if value & 0x800000:
        value -= 1 << 24
    return value


def sign_extend(value: int, bit_width: int) -> int:
    sign_bit = 1 << (bit_width - 1)
    mask = (1 << bit_width) - 1
    value &= mask
    return value - (1 << bit_width) if (value & sign_bit) else value


def parse_delta_frame(delta_bytes: bytes, channels: int, bit_width: int, total_bit_length: int) -> list[list[int]]:
    bitset: list[int] = []
    for byte in delta_bytes:
        for i in range(8):
            bitset.append((byte >> i) & 0x01)
    samples: list[list[int]] = []
    offset = 0
    while offset < total_bit_length:
        channel_samples: list[int] = []
        for _ in range(channels):
            value = 0
            for i in range(bit_width):
                value |= bitset[offset + i] << i
            channel_samples.append(sign_extend(value, bit_width))
            offset += bit_width
        samples.append(channel_samples)
    return samples


def parse_delta_frame_ref_samples(payload: bytes, channels: int, resolution_bits: int) -> list[int]:
    resolution_bytes = (resolution_bits + 7) // 8
    out: list[int] = []
    offset = 0
    for _ in range(channels):
        raw_value = int.from_bytes(payload[offset: offset + resolution_bytes], byteorder="little", signed=False)
        out.append(sign_extend(raw_value, resolution_bits))
        offset += resolution_bytes
    return out


def parse_delta_frames_all(payload: bytes, channels: int, resolution_bits: int) -> list[list[int]]:
    ref = parse_delta_frame_ref_samples(payload, channels, resolution_bits)
    offset = channels * ((resolution_bits + 7) // 8)
    samples: list[list[int]] = [ref]
    while offset < len(payload):
        if offset + 2 > len(payload):
            break
        delta_size = payload[offset]
        sample_count = payload[offset + 1]
        offset += 2
        bit_length = sample_count * delta_size * channels
        byte_length = (bit_length + 7) // 8
        if offset + byte_length > len(payload):
            break
        delta_payload = payload[offset: offset + byte_length]
        offset += byte_length
        delta_samples = parse_delta_frame(delta_payload, channels, delta_size, bit_length)
        for delta in delta_samples:
            previous = samples[-1]
            samples.append([previous[i] + delta[i] for i in range(channels)])
    return samples


def parse_pmd_data_frame(raw: bytes):
    if len(raw) < 10:
        return None
    measurement_type = raw[0] & 0x3F
    timestamp = int.from_bytes(raw[1:9], byteorder="little", signed=False)
    frame_type_field = raw[9]
    compressed = (frame_type_field & 0x80) != 0
    frame_type = frame_type_field & 0x7F
    content = raw[10:]
    return {
        "measurement_type": measurement_type,
        "timestamp": timestamp,
        "frame_type": frame_type,
        "compressed": compressed,
        "content": content,
    }


def parse_ppg_type0_uncompressed(content: bytes):
    sample_size = 12
    samples = []
    i = 0
    while i + sample_size <= len(content):
        ch0 = to_signed24_le(content[i], content[i + 1], content[i + 2])
        ch1 = to_signed24_le(content[i + 3], content[i + 4], content[i + 5])
        ch2 = to_signed24_le(content[i + 6], content[i + 7], content[i + 8])
        ambient = to_signed24_le(content[i + 9], content[i + 10], content[i + 11])
        samples.append((ch0, ch1, ch2, ambient))
        i += sample_size
    return samples


def parse_ppg_type0_compressed(content: bytes) -> list[tuple[int, int, int, int]]:
    decoded = parse_delta_frames_all(content, channels=4, resolution_bits=24)
    return [(s[0], s[1], s[2], s[3]) for s in decoded]


def pick_supported_or_fallback(supported: list[int], requested: int) -> int:
    if not supported:
        return requested
    if requested in supported:
        return requested
    lower_or_equal = [v for v in supported if v <= requested]
    if lower_or_equal:
        return max(lower_or_equal)
    return min(supported)


def cp_status_name(status: int) -> str:
    return CP_STATUS_NAMES.get(status, f"UNKNOWN_STATUS_{status}")


# ─────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VeritySample:
    elapsed: float
    ppg0: int
    ppg1: int
    ppg2: int
    ambient: int


@dataclass
class AM801Sample:
    host_timestamp_ns: int
    host_timestamp_iso: str
    elapsed: float
    red: int
    infrared: int
    background: int


# ─────────────────────────────────────────────────────────────────────
#  Trim window helper
# ─────────────────────────────────────────────────────────────────────

def trim_window(values: list[float], window_seconds: float) -> list[float]:
    if window_seconds <= 0 or not values:
        return values
    cutoff = values[-1] - window_seconds
    for index, value in enumerate(values):
        if value >= cutoff:
            return values[index:]
    return values


# ─────────────────────────────────────────────────────────────────────
#  AM801 helpers  (from  am801_raw_ppg_plot.py)
# ─────────────────────────────────────────────────────────────────────

def extract_raw_sample(packet: proto.Packet, start_wall_time: float) -> AM801Sample | None:
    if not packet.valid_checksum or packet.command != proto.CMD_RAW or len(packet.payload) != 9:
        return None
    red = proto.u24_be(packet.payload[0:3])
    infrared = proto.u24_be(packet.payload[3:6])
    background = proto.u24_be(packet.payload[6:9])
    host_timestamp_ns = time.time_ns()
    host_timestamp_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
    elapsed = (host_timestamp_ns / 1_000_000_000) - start_wall_time
    return AM801Sample(
        host_timestamp_ns=host_timestamp_ns,
        host_timestamp_iso=host_timestamp_iso,
        elapsed=elapsed,
        red=red,
        infrared=infrared,
        background=background,
    )


def capture_am801_samples(
    ser,
    stop_event: threading.Event,
    start_wall_time: float,
    samples: deque,
    lock: threading.Lock,
    csv_writer,
) -> None:
    buffer = bytearray()
    raw_packet_index = 0
    while not stop_event.is_set():
        chunk = ser.read(256)
        if not chunk:
            continue
        chunk_host_ts_ns = time.time_ns()
        buffer.extend(chunk)
        for packet in proto.parse_packets(buffer):
            sample = extract_raw_sample(packet, start_wall_time)
            if sample is None:
                continue
            # Spread timestamps across packet bursts by giving each raw packet
            # a unique nanosecond timestamp (nanosecond-level resolution)
            ts_ns = chunk_host_ts_ns + raw_packet_index
            raw_packet_index += 1
            ts_sec = ts_ns / 1_000_000_000
            ts_iso = datetime.fromtimestamp(ts_sec).astimezone().isoformat(timespec="milliseconds")
            elapsed = ts_sec - start_wall_time
            sample.host_timestamp_ns = ts_ns
            sample.host_timestamp_iso = ts_iso
            sample.elapsed = elapsed
            with lock:
                samples.append(sample)
            if csv_writer is not None:
                csv_writer.writerow({
                    "host_timestamp_ns": sample.host_timestamp_ns,
                    "host_timestamp_iso": sample.host_timestamp_iso,
                    "elapsed_s": f"{sample.elapsed:.3f}",
                    "red": sample.red,
                    "infrared": sample.infrared,
                    "background": sample.background,
                })


def detect_ch341_port() -> str:
    for port in list_ports.comports():
        description = (port.description or "").lower()
        manufacturer = (port.manufacturer or "").lower()
        hwid = (port.hwid or "").lower()
        # Windows-style string match
        if "usb-serial ch341" in description or "wch" in manufacturer or "ch341" in hwid:
            return port.device
        # Linux-style match: CH340/CH341 uses WCH's vendor ID 1A86
        if getattr(port, "vid", None) == 0x1A86:
            return port.device
    detected = [f"{port.device} ({port.description})" for port in list_ports.comports()]
    detected_text = ", ".join(detected) if detected else "none"
    raise RuntimeError(
        "Could not auto-detect a USB-SERIAL CH341 port. "
        f"Use --am801-port explicitly. Detected ports: {detected_text}"
    )

# ─────────────────────────────────────────────────────────────────────
#  Main dual-device application
# ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Dual PPG Plotter: Verity Sense (BLE) + AM801 (serial)")
    # Verity Sense args
    parser.add_argument("--verity-address", help="BLE MAC address for Verity Sense")
    parser.add_argument("--verity-name", help="Name hint for Verity Sense scanner")
    parser.add_argument("--sample-rate", type=int, default=55, help="PPG sample rate target in Hz")
    parser.add_argument("--resolution", type=int, default=22, help="PPG resolution to request")
    # AM801 args
    parser.add_argument("--am801-port", help="Serial port for AM801 (e.g. COM6). Auto-detect if omitted")
    parser.add_argument("--am801-baud", type=int, default=230400)
    # Common args
    parser.add_argument("--window-seconds", type=float, default=20.0, help="Visible plot window in seconds")
    parser.add_argument("--countdown", type=int, default=5, help="Countdown seconds after both devices connected")
    parser.add_argument("--verity-output", help="CSV output path for Verity Sense (default: auto-timestamped)")
    parser.add_argument("--am801-output", help="CSV output path for AM801 (default: auto-timestamped)")
    parser.add_argument("--show-raw-hex", action="store_true", help="Print Verity PMD data packet hex")
    args = parser.parse_args()

    # Default CSV paths
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    verity_csv_path = args.verity_output or f"verity_sense_ppg_{timestamp_str}.csv"
    am801_csv_path = args.am801_output or f"am801_raw_ppg_{timestamp_str}.csv"

    # Shared state
    verity_samples: deque[VeritySample] = deque(maxlen=12000)
    am801_samples: deque[AM801Sample] = deque(maxlen=5000)
    am801_lock = threading.Lock()

    am801_csv_file = None
    am801_csv_writer = None
    # Verity CSV: write incrementally to a temporary file, rename on success
    verity_csv_tmp_path = verity_csv_path + ".tmp"
    verity_csv_file = open(verity_csv_tmp_path, "w", newline="", encoding="utf-8")
    verity_csv_writer = csv.DictWriter(verity_csv_file, fieldnames=[
        "host_timestamp_ns", "host_timestamp_iso", "pmd_timestamp",
        "ppg0", "ppg1", "ppg2", "ambient",
    ])
    verity_csv_writer.writeheader()
    verity_csv_file.flush()
    print(f"Verity CSV (temp): {Path(verity_csv_tmp_path).resolve()}")
    verity_save_ok = False

    stop_event = threading.Event()
    start_wall_time = time.time()
    both_connected_time: float | None = None
    countdown_remaining: int = args.countdown
    countdown_done = threading.Event()
    recording = False

    # AM801 connection
    am801_port = args.am801_port or detect_ch341_port()
    am801_ser = proto.open_serial(am801_port, args.am801_baud)
    print(f"AM801: Opened {am801_port} @ {args.am801_baud} baud")

    # Send AM801 start command
    proto.send_command(am801_ser, proto.CMD_CONTROL, b"\x01")
    print("AM801: Start command sent")

    # Open AM801 CSV
    am801_csv_file = open(am801_csv_path, "w", newline="", encoding="utf-8")
    am801_csv_writer = csv.DictWriter(
        am801_csv_file,
        fieldnames=["host_timestamp_ns", "host_timestamp_iso", "elapsed_s", "red", "infrared", "background"],
    )
    am801_csv_writer.writeheader()
    print(f"AM801 CSV: {Path(am801_csv_path).resolve()}")

    # Flush AM801 CSV periodically
    am801_csv_file.flush()

    # Start AM801 capture thread (immediately recording to CSV)
    am801_thread = threading.Thread(
        target=capture_am801_samples,
        args=(am801_ser, stop_event, start_wall_time, am801_samples, am801_lock, am801_csv_writer),
        daemon=True,
    )
    am801_thread.start()

    # Verity Sense connection
    print("Verity: Scanning for Polar Verity Sense...")
    verity_address = await find_device(args.verity_address, args.verity_name)

    cp = PmdController()
    verity_active_sample_rate = {"value": args.sample_rate}
    verity_stats = {"type0_samples": 0, "compressed_type0_frames": 0, "other_frames": 0}
    # Running counter for monotonically increasing timestamps (avoids BLE jitter)
    verity_sample_counter = 0
    start_wall_time_ns = time.time_ns()

    def verity_pmd_data_handler(_sender, data: bytearray):
        nonlocal verity_sample_counter
        raw = bytes(data)
        frame = parse_pmd_data_frame(raw)
        if frame is None:
            return
        if frame["measurement_type"] != PPG_MEASUREMENT_TYPE:
            return
        if args.show_raw_hex:
            print("PMD_DATA:", raw.hex(" "))

        if frame["frame_type"] == 0:
            if frame["compressed"]:
                samples = parse_ppg_type0_compressed(frame["content"])
                verity_stats["compressed_type0_frames"] += 1
            else:
                samples = parse_ppg_type0_uncompressed(frame["content"])
            sample_interval_ns = int(1_000_000_000 / max(1, verity_active_sample_rate["value"]))
            for s in samples:
                # Use a running counter to generate perfectly evenly spaced timestamps,
                # independent of BLE notification arrival jitter
                sample_host_ts_ns = start_wall_time_ns + int(verity_sample_counter * sample_interval_ns)
                verity_sample_counter += 1
                sample_host_ts_sec = sample_host_ts_ns / 1_000_000_000
                sample_host_ts_iso = datetime.fromtimestamp(sample_host_ts_sec).astimezone().isoformat(timespec="milliseconds")
                elapsed = sample_host_ts_sec - start_wall_time
                verity_csv_writer.writerow({
                    "host_timestamp_ns": sample_host_ts_ns,
                    "host_timestamp_iso": sample_host_ts_iso,
                    "pmd_timestamp": frame["timestamp"],
                    "ppg0": s[0],
                    "ppg1": s[1],
                    "ppg2": s[2],
                    "ambient": s[3],
                })
                verity_samples.append(VeritySample(elapsed=elapsed, ppg0=s[0], ppg1=s[1], ppg2=s[2], ambient=s[3]))
                verity_stats["type0_samples"] += 1
                print(f"Verity: ts={frame['timestamp']} ppg={s[0]},{s[1]},{s[2]} ambient={s[3]}")
        else:
            verity_stats["other_frames"] += 1

    async with BleakClient(verity_address) as client:
        print("Verity: Connected:", client.is_connected)
        await client.start_notify(PMD_CP, cp.cp_notification_handler)
        await client.start_notify(PMD_DATA, verity_pmd_data_handler)

        # Query and start PPG
        status, settings_blob = await cp.send_cp_command(client, GET_MEASUREMENT_SETTINGS, bytes([PPG_MEASUREMENT_TYPE]))
        if status != 0:
            raise RuntimeError(f"GET_MEASUREMENT_SETTINGS failed with status={status}")
        settings = parse_settings_blob(settings_blob)
        print("Verity: Available PPG settings:", settings)

        if args.sample_rate > max(settings.get(SETTING_SAMPLE_RATE, [0])) or args.sample_rate >= 135:
            sdk_status, _ = await cp.send_cp_command(client, REQUEST_MEASUREMENT_START, bytes([SDK_MODE_MEASUREMENT_TYPE]))
            if sdk_status in (0, 6):
                state = "enabled" if sdk_status == 0 else "already enabled"
                print(f"Verity: SDK mode {state}.")
                full_status, full_blob = await cp.send_cp_command(client, GET_SDK_MODE_MEASUREMENT_SETTINGS, bytes([PPG_MEASUREMENT_TYPE]))
                if full_status == 0:
                    settings = parse_settings_blob(full_blob)
                    print("Verity: Available PPG settings (SDK mode):", settings)
                else:
                    print(f"Verity: SDK settings query failed, continuing")
            else:
                print(f"Verity: Failed to enable SDK mode, continuing")

        sample_rate = args.sample_rate
        resolution = args.resolution
        if SETTING_SAMPLE_RATE in settings and sample_rate not in settings[SETTING_SAMPLE_RATE]:
            sample_rate = pick_supported_or_fallback(settings[SETTING_SAMPLE_RATE], sample_rate)
            print(f"Verity: Using sample rate {sample_rate}")
        if SETTING_RESOLUTION in settings and resolution not in settings[SETTING_RESOLUTION]:
            resolution = pick_supported_or_fallback(settings[SETTING_RESOLUTION], resolution)
            print(f"Verity: Using resolution {resolution}")

        verity_active_sample_rate["value"] = sample_rate
        selected_tlv = build_selected_settings_tlv(settings, sample_rate, resolution)
        start_params = bytes([PPG_MEASUREMENT_TYPE]) + selected_tlv

        status, start_resp = await cp.send_cp_command(client, REQUEST_MEASUREMENT_START, start_params)
        if status != 0:
            print(f"Verity: Start rejected ({cp_status_name(status)}), retrying...")
            sample_res_tlv = bytearray()
            if SETTING_SAMPLE_RATE in settings:
                _append_setting_tlv(sample_res_tlv, SETTING_SAMPLE_RATE, sample_rate)
            if SETTING_RESOLUTION in settings:
                _append_setting_tlv(sample_res_tlv, SETTING_RESOLUTION, resolution)
            status, start_resp = await cp.send_cp_command(client, REQUEST_MEASUREMENT_START,
                                                           bytes([PPG_MEASUREMENT_TYPE]) + bytes(sample_res_tlv))
        if status != 0:
            print(f"Verity: Retry rejected, trying with defaults...")
            status, start_resp = await cp.send_cp_command(client, REQUEST_MEASUREMENT_START,
                                                           bytes([PPG_MEASUREMENT_TYPE]))
        if status != 0:
            raise RuntimeError(f"Verity: Could not start PPG: status={status}")
        print(f"Verity: PPG started @ {sample_rate} Hz, {resolution}-bit")

        # Both devices are now connected and streaming
        both_connected_time = time.time()
        print(f"Both devices connected. Countdown: {args.countdown} seconds...")

        # ── Build combined plot ──────────────────────────────────────────
        plt.style.use("seaborn-v0_8-darkgrid")
        figure = plt.figure(figsize=(16, 10))
        figure.suptitle(f"Dual PPG: Verity Sense + AM801", fontsize=14)

        # Verity subplots (4 rows, 1 col on left)
        verity_axes = []
        verity_lines = []
        verity_colors = [("ppg0", "#d32f2f"), ("ppg1", "#f57c00"), ("ppg2", "#388e3c"), ("ambient", "#455a64")]
        for i, (label, color) in enumerate(verity_colors):
            ax = figure.add_subplot(7, 1, i + 1)
            (line,) = ax.plot([], [], color=color, linewidth=1.2)
            ax.set_ylabel(f"Verity {label}")
            ax.grid(True, alpha=0.25)
            verity_axes.append(ax)
            verity_lines.append(line)

        # AM801 subplots (3 rows below)
        am801_axes = []
        am801_lines = []
        am801_colors = [("red", "#d32f2f"), ("infrared", "#f57c00"), ("background", "#455a64")]
        for i, (label, color) in enumerate(am801_colors):
            ax = figure.add_subplot(7, 1, 5 + i)
            (line,) = ax.plot([], [], color=color, linewidth=1.2)
            ax.set_ylabel(f"AM801 {label}")
            ax.grid(True, alpha=0.25)
            am801_axes.append(ax)
            am801_lines.append(line)

        am801_axes[-1].set_xlabel("Elapsed time (s)")
        figure.tight_layout(rect=(0, 0, 1, 0.97))

        # ── Plot / countdown loop ──────────────────────────────────────
        try:
            while plt.get_fignums():
                # Update countdown timer in title
                current_time = time.time()
                if not countdown_done.is_set():
                    elapsed_since_connected = current_time - both_connected_time
                    remaining = max(0, args.countdown - int(elapsed_since_connected))
                    countdown_remaining = remaining
                    figure.suptitle(
                        f"Dual PPG: Verity Sense + AM801  |  Recording starts in {remaining}s",
                        fontsize=14,
                    )
                    if remaining <= 0:
                        countdown_done.set()
                        recording = True
                        figure.suptitle(
                            f"Dual PPG: Verity Sense + AM801  |  RECORDING (Ctrl+C to stop)",
                            fontsize=14,
                        )
                        print("Countdown finished. Recording...")
                else:
                    if recording:
                        elapsed_rec = current_time - both_connected_time - args.countdown
                        figure.suptitle(
                            f"Dual PPG: Verity Sense + AM801  |  Recording: {elapsed_rec:.0f}s (Ctrl+C to stop)",
                            fontsize=14,
                        )

                # Update Verity plot
                verity_snapshot = list(verity_samples)
                if verity_snapshot:
                    times_v = [s.elapsed for s in verity_snapshot]
                    ppg0_v = [s.ppg0 for s in verity_snapshot]
                    ppg1_v = [s.ppg1 for s in verity_snapshot]
                    ppg2_v = [s.ppg2 for s in verity_snapshot]
                    ambient_v = [s.ambient for s in verity_snapshot]

                    # Apply time window: keep only last window_seconds of data
                    times_v = trim_window(times_v, args.window_seconds)
                    if args.window_seconds > 0 and times_v:
                        cutoff = times_v[0]
                        idx = next((i for i, v in enumerate([s.elapsed for s in verity_snapshot]) if v >= cutoff), 0)
                        ppg0_v = ppg0_v[idx:]
                        ppg1_v = ppg1_v[idx:]
                        ppg2_v = ppg2_v[idx:]
                        ambient_v = ambient_v[idx:]

                    for ax, line, values in zip(verity_axes, verity_lines, [ppg0_v, ppg1_v, ppg2_v, ambient_v]):
                        line.set_data(times_v, values)
                        if times_v:
                            # Use a continuous scrolling window anchored to current time
                            window_left = max(times_v[-1] - args.window_seconds, 0)
                            ax.set_xlim(window_left, max(times_v[-1], 0.1))
                            lo = min(values)
                            hi = max(values)
                            if lo == hi:
                                ax.set_ylim(lo - 1, hi + 1)
                            else:
                                pad = max(1, int((hi - lo) * 0.05))
                                ax.set_ylim(lo - pad, hi + pad)

                # Update AM801 plot
                with am801_lock:
                    am801_snapshot = list(am801_samples)
                if am801_snapshot:
                    times_a = [s.elapsed for s in am801_snapshot]
                    red_a = [s.red for s in am801_snapshot]
                    infrared_a = [s.infrared for s in am801_snapshot]
                    background_a = [s.background for s in am801_snapshot]

                    # Apply time window: keep only last window_seconds of data
                    times_a = trim_window(times_a, args.window_seconds)
                    if args.window_seconds > 0 and times_a:
                        cutoff = times_a[0]
                        idx = next((i for i, v in enumerate([s.elapsed for s in am801_snapshot]) if v >= cutoff), 0)
                        red_a = red_a[idx:]
                        infrared_a = infrared_a[idx:]
                        background_a = background_a[idx:]

                    for ax, line, values in zip(am801_axes, am801_lines, [red_a, infrared_a, background_a]):
                        line.set_data(times_a, values)
                        if times_a:
                            # Use a continuous scrolling window anchored to current time
                            window_left = max(times_a[-1] - args.window_seconds, 0)
                            ax.set_xlim(window_left, max(times_a[-1], 0.1))
                            lo = min(values)
                            hi = max(values)
                            if lo == hi:
                                ax.set_ylim(lo - 1, hi + 1)
                            else:
                                pad = max(1, int((hi - lo) * 0.05))
                                ax.set_ylim(lo - pad, hi + pad)

                figure.canvas.draw_idle()
                plt.pause(0.05)
                await asyncio.sleep(0.05)

        except KeyboardInterrupt:
            print("\nCtrl+C received, stopping...")
        finally:
            stop_event.set()
            if plt.get_fignums():
                plt.close(figure)

            # Stop Verity Sense
            try:
                status, _ = await cp.send_cp_command(client, STOP_MEASUREMENT, bytes([PPG_MEASUREMENT_TYPE]))
                print("Verity: STOP status:", status)
            except Exception as exc:
                print(f"Verity: STOP failed: {exc}")

            await client.stop_notify(PMD_DATA)
            await client.stop_notify(PMD_CP)

            # Stop AM801
            try:
                proto.send_command(am801_ser, proto.CMD_CONTROL, b"\x00")
                print("AM801: Stop command sent")
            except Exception as exc:
                print(f"AM801: Stop failed: {exc}")

            am801_thread.join(timeout=2.0)

            # Close AM801 CSV and flush
            if am801_csv_file is not None:
                am801_csv_file.flush()
                am801_csv_file.close()
                print(f"AM801 CSV saved: {am801_csv_path} ({len(am801_samples)} in-memory samples)")

            # Close Verity CSV temp file and rename to final path
            if verity_csv_file is not None:
                verity_csv_file.flush()
                verity_csv_file.close()
                # Rename temp file to final CSV path
                try:
                    Path(verity_csv_tmp_path).replace(verity_csv_path)
                    verity_save_ok = True
                    print(f"Verity CSV saved: {verity_csv_path}")
                except Exception as exc:
                    print(f"Verity CSV rename failed: {exc}. File remains at: {verity_csv_tmp_path}")

    # Final summary
    if not verity_save_ok:
        print("Verity: Warning - CSV may be incomplete due to rename failure.")


async def find_device(address: str | None, name_hint: str | None):
    if address:
        return address
    print("Scanning for Polar Verity Sense...")
    try:
        devices = await BleakScanner.discover(timeout=8.0)
    except OSError as exc:
        if platform.system() == "Windows":
            raise RuntimeError(
                "BLE scan failed on Windows. Make sure Bluetooth is ON, Verity Sense is awake/advertising. "
                f"Original error: {exc}"
            ) from exc
        raise
    candidates = []
    for d in devices:
        n = (d.name or "").lower()
        if "polar" in n or "verity" in n:
            candidates.append(d)
    if name_hint:
        hint = name_hint.lower()
        for d in candidates:
            if d.name and hint in d.name.lower():
                return d.address
    if not candidates:
        raise RuntimeError("No Polar-like BLE device found. Use --verity-address to force one.")
    print("Using:", candidates[0].name, candidates[0].address)
    return candidates[0].address


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc