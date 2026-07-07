import asyncio
import argparse
import csv
import platform
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
from bleak import BleakClient, BleakScanner

PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CP = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

# PMD control point opcodes
GET_MEASUREMENT_SETTINGS = 0x01
REQUEST_MEASUREMENT_START = 0x02
STOP_MEASUREMENT = 0x03
GET_SDK_MODE_MEASUREMENT_SETTINGS = 0x04

# Measurement type id (PmdMeasurementType.PPG)
PPG_MEASUREMENT_TYPE = 0x01
SDK_MODE_MEASUREMENT_TYPE = 0x09

# PmdSettingType ids
SETTING_SAMPLE_RATE = 0x00
SETTING_RESOLUTION = 0x01
SETTING_RANGE = 0x02
SETTING_RANGE_MILLIUNIT = 0x03
SETTING_CHANNELS = 0x04
SETTING_FACTOR = 0x05
SETTING_SECURITY = 0x06

# Setting field sizes in bytes
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

        # Response format in SDK: [0]=0xF0, [1]=opcode, [2]=measurementType, [3]=status, [4]=more, [5:]=params
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
            # only control point response packets
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
            chunk = blob[i : i + field_size]
            i += field_size
            if field_size == 1:
                vals.append(chunk[0])
            elif field_size in (2, 4):
                vals.append(int.from_bytes(chunk, byteorder="little", signed=False))
            elif field_size == 16:
                # security blobs are raw bytes, represent as int list for visibility
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
    # Build TLV only for settings reported by the device.
    out = bytearray()

    if SETTING_SAMPLE_RATE in settings:
        _append_setting_tlv(out, SETTING_SAMPLE_RATE, sample_rate)

    if SETTING_RESOLUTION in settings:
        _append_setting_tlv(out, SETTING_RESOLUTION, resolution)

    # Some firmwares require channel count to be present as well.
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
        raw_value = int.from_bytes(payload[offset : offset + resolution_bytes], byteorder="little", signed=False)
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

        delta_payload = payload[offset : offset + byte_length]
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
    # SDK type 0 sample = 4 channels * 3 bytes signed each
    # ch0, ch1, ch2, ambient
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
    # Mirror SDK parser: 4 channels, 24-bit signed reference and delta coding.
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


@dataclass
class VeritySample:
    elapsed: float
    ppg0: int
    ppg1: int
    ppg2: int
    ambient: int


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
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 9))
    lines = []
    colors = [
        ("ppg0", "#d32f2f"),
        ("ppg1", "#f57c00"),
        ("ppg2", "#388e3c"),
        ("ambient", "#455a64"),
    ]
    for axis, (label, color) in zip(axes, colors, strict=True):
        (line,) = axis.plot([], [], color=color, linewidth=1.2)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        lines.append(line)
    axes[-1].set_xlabel("Elapsed time (s)")
    figure.tight_layout()
    return figure, list(axes), lines


async def find_device(address: str | None, name_hint: str | None):
    if address:
        return address

    print("Scanning for Polar Verity Sense...")
    try:
        devices = await BleakScanner.discover(timeout=8.0)
    except OSError as exc:
        if platform.system() == "Windows":
            raise RuntimeError(
                "BLE scan failed on Windows (device not ready). "
                "Make sure Bluetooth is ON, Verity Sense is awake/advertising, and no other app is locking the adapter. "
                "Then retry, or pass --address to skip scanning. "
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
        raise RuntimeError("No Polar-like BLE device found. Use --address to force one.")

    print("Using:", candidates[0].name, candidates[0].address)
    return candidates[0].address


async def main():
    parser = argparse.ArgumentParser(description="Read raw PPG PMD stream from Polar Verity Sense")
    parser.add_argument("--address", help="BLE MAC address (recommended on Windows)")
    parser.add_argument("--name", help="Name hint for scanner selection")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=55,
        help="PPG sample rate target in Hz (will be clamped to a supported value)",
    )
    parser.add_argument("--resolution", type=int, default=22, help="PPG resolution to request")
    parser.add_argument(
        "--output",
        default=f"ppg_{platform.system().lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="CSV output path for stored PPG samples",
    )
    parser.add_argument("--window-seconds", type=float, default=20.0, help="Visible plot window in seconds")
    parser.add_argument("--show-raw-hex", action="store_true", help="Print full PMD data packet hex")
    args = parser.parse_args()

    address = await find_device(args.address, args.name)
    cp = PmdController()
    collected_rows = []
    plot_samples: deque[VeritySample] = deque(maxlen=12000)
    start_wall_time = time.time()
    active_sample_rate = {"value": args.sample_rate}
    stats = {
        "type0_samples": 0,
        "compressed_type0_frames": 0,
        "other_frames": 0,
    }

    def pmd_data_handler(_sender, data: bytearray):
        raw = bytes(data)
        host_ts_ns = time.time_ns()
        frame = parse_pmd_data_frame(raw)
        if frame is None:
            return

        if frame["measurement_type"] != PPG_MEASUREMENT_TYPE:
            return

        if args.show_raw_hex:
            print("PMD_DATA:", raw.hex(" "))

        # Decode PPG type-0 from both raw and compressed PMD frames.
        if frame["frame_type"] == 0:
            if frame["compressed"]:
                samples = parse_ppg_type0_compressed(frame["content"])
                stats["compressed_type0_frames"] += 1
            else:
                samples = parse_ppg_type0_uncompressed(frame["content"])

            for index, s in enumerate(samples):
                per_sample_offset_ns = int(index * (1_000_000_000 / max(1, active_sample_rate["value"])))
                sample_host_ts_ns = host_ts_ns + per_sample_offset_ns
                sample_host_ts_sec = sample_host_ts_ns / 1_000_000_000
                sample_host_ts_iso = datetime.fromtimestamp(sample_host_ts_sec).astimezone().isoformat(timespec="milliseconds")
                elapsed = sample_host_ts_sec - start_wall_time

                collected_rows.append(
                    {
                        "host_timestamp_ns": sample_host_ts_ns,
                        "host_timestamp_iso": sample_host_ts_iso,
                        "pmd_timestamp": frame["timestamp"],
                        "ppg0": s[0],
                        "ppg1": s[1],
                        "ppg2": s[2],
                        "ambient": s[3],
                    }
                )
                plot_samples.append(
                    VeritySample(
                        elapsed=elapsed,
                        ppg0=s[0],
                        ppg1=s[1],
                        ppg2=s[2],
                        ambient=s[3],
                    )
                )
                stats["type0_samples"] += 1
                print(f"ts={frame['timestamp']} type0 ppg={s[0]},{s[1]},{s[2]} ambient={s[3]}")
        else:
            stats["other_frames"] += 1
            print(
                f"ts={frame['timestamp']} frameType={frame['frame_type']} "
                f"compressed={frame['compressed']} payloadBytes={len(frame['content'])}"
            )

    async with BleakClient(address) as client:
        print("Connected:", client.is_connected)

        await client.start_notify(PMD_CP, cp.cp_notification_handler)
        await client.start_notify(PMD_DATA, pmd_data_handler)

        # Query available settings for PPG (opcode 0x01, measurement type byte 0x01 for ONLINE PPG)
        status, settings_blob = await cp.send_cp_command(client, GET_MEASUREMENT_SETTINGS, bytes([PPG_MEASUREMENT_TYPE]))
        if status != 0:
            raise RuntimeError(f"GET_MEASUREMENT_SETTINGS failed with status={status}")

        settings = parse_settings_blob(settings_blob)
        print("Available PPG settings (normal mode):", settings)

        # High-rate PPG settings are available in SDK mode for Verity Sense.
        if (
            args.sample_rate > max(settings.get(SETTING_SAMPLE_RATE, [0]))
            or args.sample_rate >= 135
        ):
            sdk_status, _ = await cp.send_cp_command(client, REQUEST_MEASUREMENT_START, bytes([SDK_MODE_MEASUREMENT_TYPE]))
            if sdk_status in (0, 6):
                state = "enabled" if sdk_status == 0 else "already enabled"
                print(f"SDK mode {state}.")
                full_status, full_blob = await cp.send_cp_command(
                    client,
                    GET_SDK_MODE_MEASUREMENT_SETTINGS,
                    bytes([PPG_MEASUREMENT_TYPE]),
                )
                if full_status == 0:
                    settings = parse_settings_blob(full_blob)
                    print("Available PPG settings (SDK mode):", settings)
                else:
                    print(
                        "GET_SDK_MODE_MEASUREMENT_SETTINGS failed with "
                        f"status={full_status} ({cp_status_name(full_status)}), "
                        "continuing with normal-mode settings"
                    )
            else:
                print(
                    "Failed to enable SDK mode: "
                    f"status={sdk_status} ({cp_status_name(sdk_status)}). "
                    "Continuing with normal-mode settings."
                )

        if SETTING_SAMPLE_RATE in settings:
            print("Supported sample rates:", sorted(settings[SETTING_SAMPLE_RATE]))
        if SETTING_RESOLUTION in settings:
            print("Supported resolutions:", sorted(settings[SETTING_RESOLUTION]))

        sample_rate = args.sample_rate
        resolution = args.resolution

        if SETTING_SAMPLE_RATE in settings and sample_rate not in settings[SETTING_SAMPLE_RATE]:
            sample_rate = pick_supported_or_fallback(settings[SETTING_SAMPLE_RATE], sample_rate)
            print(f"Requested sample rate not supported, using closest supported value: {sample_rate}")
        active_sample_rate["value"] = sample_rate

        if SETTING_RESOLUTION in settings and resolution not in settings[SETTING_RESOLUTION]:
            resolution = pick_supported_or_fallback(settings[SETTING_RESOLUTION], resolution)
            print(f"Requested resolution not supported, using closest supported value: {resolution}")

        selected_tlv = build_selected_settings_tlv(settings, sample_rate, resolution)
        start_params = bytes([PPG_MEASUREMENT_TYPE]) + selected_tlv

        status, start_resp = await cp.send_cp_command(client, REQUEST_MEASUREMENT_START, start_params)

        if status != 0:
            print(
                "Start with sample+resolution(+channels) rejected: "
                f"status={status} ({cp_status_name(status)}), retrying with sample+resolution only..."
            )
            sample_res_tlv = bytearray()
            if SETTING_SAMPLE_RATE in settings:
                _append_setting_tlv(sample_res_tlv, SETTING_SAMPLE_RATE, sample_rate)
            if SETTING_RESOLUTION in settings:
                _append_setting_tlv(sample_res_tlv, SETTING_RESOLUTION, resolution)
            status, start_resp = await cp.send_cp_command(
                client,
                REQUEST_MEASUREMENT_START,
                bytes([PPG_MEASUREMENT_TYPE]) + bytes(sample_res_tlv),
            )

        if status != 0:
            print(
                "Start with sample+resolution rejected: "
                f"status={status} ({cp_status_name(status)}), retrying with sample-rate only..."
            )
            sample_only_tlv = bytearray()
            if SETTING_SAMPLE_RATE in settings:
                _append_setting_tlv(sample_only_tlv, SETTING_SAMPLE_RATE, sample_rate)
            status, start_resp = await cp.send_cp_command(
                client,
                REQUEST_MEASUREMENT_START,
                bytes([PPG_MEASUREMENT_TYPE]) + bytes(sample_only_tlv),
            )

        if status != 0:
            # Final fallback: firmware-selected default settings.
            print(
                "Start with explicit TLVs still rejected: "
                f"status={status} ({cp_status_name(status)}), retrying with default device settings..."
            )
            status, start_resp = await cp.send_cp_command(client, REQUEST_MEASUREMENT_START, bytes([PPG_MEASUREMENT_TYPE]))

        if status != 0:
            raise RuntimeError(
                "REQUEST_MEASUREMENT_START failed with "
                f"status={status} ({cp_status_name(status)}), params={start_resp.hex(' ')}"
            )

        print(f"PPG started. sample_rate={sample_rate} resolution={resolution}")
        print("Streaming until Ctrl+C or plot window close...")

        figure, axes, lines = build_plot()
        figure.suptitle(f"Verity Sense raw PPG on {address} @ {sample_rate} Hz")

        try:
            while plt.get_fignums():
                snapshot = list(plot_samples)
                if snapshot:
                    times = [sample.elapsed for sample in snapshot]
                    ppg0_values = [sample.ppg0 for sample in snapshot]
                    ppg1_values = [sample.ppg1 for sample in snapshot]
                    ppg2_values = [sample.ppg2 for sample in snapshot]
                    ambient_values = [sample.ambient for sample in snapshot]

                    times = trim_window(times, args.window_seconds)
                    if args.window_seconds > 0 and times:
                        cutoff = times[0]
                        start_index = 0
                        for index, value in enumerate([sample.elapsed for sample in snapshot]):
                            if value >= cutoff:
                                start_index = index
                                break
                        ppg0_values = ppg0_values[start_index:]
                        ppg1_values = ppg1_values[start_index:]
                        ppg2_values = ppg2_values[start_index:]
                        ambient_values = ambient_values[start_index:]

                    datasets = [ppg0_values, ppg1_values, ppg2_values, ambient_values]
                    for axis, line, values in zip(axes, lines, datasets, strict=True):
                        line.set_data(times, values)
                        if times and values:
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
                await asyncio.sleep(0.05)
        except KeyboardInterrupt:
            print("Ctrl+C received, stopping stream...")
        finally:
            if plt.get_fignums():
                plt.close(figure)

            try:
                status, _ = await cp.send_cp_command(client, STOP_MEASUREMENT, bytes([PPG_MEASUREMENT_TYPE]))
                print("STOP_MEASUREMENT status:", status)
            except Exception as exc:
                print(f"STOP_MEASUREMENT failed: {exc}")

            await client.stop_notify(PMD_DATA)
            await client.stop_notify(PMD_CP)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "host_timestamp_ns",
                "host_timestamp_iso",
                "pmd_timestamp",
                "ppg0",
                "ppg1",
                "ppg2",
                "ambient",
            ],
        )
        writer.writeheader()
        writer.writerows(collected_rows)

    print(f"Saved {len(collected_rows)} PPG rows to {args.output}")
    if len(collected_rows) == 0:
        print(
            "No type-0 PPG samples were decoded. "
            f"Observed compressed type-0 frames={stats['compressed_type0_frames']}, "
            f"Observed unsupported frames={stats['other_frames']}. "
            "Try running again with --show-raw-hex and share a few PMD_DATA lines."
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
