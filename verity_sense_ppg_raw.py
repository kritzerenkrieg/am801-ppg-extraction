import asyncio
import argparse
import csv
import platform
import time
from datetime import datetime
from bleak import BleakClient, BleakScanner

PMD_SERVICE = "FB005C80-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_CP = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

# PMD control point opcodes
GET_MEASUREMENT_SETTINGS = 0x01
REQUEST_MEASUREMENT_START = 0x02
STOP_MEASUREMENT = 0x03

# Measurement type id (PmdMeasurementType.PPG)
PPG_MEASUREMENT_TYPE = 0x01

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


def build_selected_settings_tlv(sample_rate: int, resolution: int) -> bytes:
    # Mirrors SDK serializeSelected(): [type][count=1][value little-endian]
    out = bytearray()

    out += bytes([SETTING_SAMPLE_RATE, 1])
    out += sample_rate.to_bytes(2, byteorder="little", signed=False)

    out += bytes([SETTING_RESOLUTION, 1])
    out += resolution.to_bytes(2, byteorder="little", signed=False)

    return bytes(out)


def to_signed24_le(b0: int, b1: int, b2: int) -> int:
    value = b0 | (b1 << 8) | (b2 << 16)
    if value & 0x800000:
        value -= 1 << 24
    return value


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


async def find_device(address: str | None, name_hint: str | None):
    if address:
        return address

    print("Scanning for Polar Verity Sense...")
    devices = await BleakScanner.discover(timeout=8.0)
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
        default=176,
        choices=[135, 176],
        help="PPG sample rate target in Hz (SDK mode): 135 or 176",
    )
    parser.add_argument("--resolution", type=int, default=22, help="PPG resolution to request")
    parser.add_argument(
        "--output",
        default=f"ppg_{platform.system().lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="CSV output path for stored PPG samples",
    )
    parser.add_argument("--show-raw-hex", action="store_true", help="Print full PMD data packet hex")
    args = parser.parse_args()

    address = await find_device(args.address, args.name)
    cp = PmdController()
    collected_rows = []

    def pmd_data_handler(_sender, data: bytearray):
        raw = bytes(data)
        host_ts_ns = time.time_ns()
        host_ts_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
        frame = parse_pmd_data_frame(raw)
        if frame is None:
            return

        if frame["measurement_type"] != PPG_MEASUREMENT_TYPE:
            return

        if args.show_raw_hex:
            print("PMD_DATA:", raw.hex(" "))

        # Parse only uncompressed type 0 directly.
        if (not frame["compressed"]) and frame["frame_type"] == 0:
            samples = parse_ppg_type0_uncompressed(frame["content"])
            for s in samples:
                collected_rows.append(
                    {
                        "host_timestamp_ns": host_ts_ns,
                        "host_timestamp_iso": host_ts_iso,
                        "pmd_timestamp": frame["timestamp"],
                        "ppg0": s[0],
                        "ppg1": s[1],
                        "ppg2": s[2],
                        "ambient": s[3],
                    }
                )
                print(f"ts={frame['timestamp']} type0 ppg={s[0]},{s[1]},{s[2]} ambient={s[3]}")
        else:
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
        print("Available PPG settings:", settings)

        sample_rate = args.sample_rate
        resolution = args.resolution

        if SETTING_SAMPLE_RATE in settings and sample_rate not in settings[SETTING_SAMPLE_RATE]:
            sample_rate = max(settings[SETTING_SAMPLE_RATE])
            print(f"Requested sample rate not supported, using {sample_rate}")

        if SETTING_RESOLUTION in settings and resolution not in settings[SETTING_RESOLUTION]:
            resolution = max(settings[SETTING_RESOLUTION])
            print(f"Requested resolution not supported, using {resolution}")

        selected_tlv = build_selected_settings_tlv(sample_rate, resolution)
        start_params = bytes([PPG_MEASUREMENT_TYPE]) + selected_tlv

        status, start_resp = await cp.send_cp_command(client, REQUEST_MEASUREMENT_START, start_params)
        if status != 0:
            raise RuntimeError(f"REQUEST_MEASUREMENT_START failed with status={status}, params={start_resp.hex(' ')}")

        print(f"PPG started. sample_rate={sample_rate} resolution={resolution}")
        print("Streaming until Ctrl+C...")

        try:
            while True:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            print("Ctrl+C received, stopping stream...")
        finally:
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


if __name__ == "__main__":
    asyncio.run(main())
