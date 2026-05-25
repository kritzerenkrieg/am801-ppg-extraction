import argparse
import time
from dataclasses import dataclass
from typing import Iterator

import serial


HEADER = b"\xFF\xFE"
DEVICE_ID = 0x23
MIN_PACKET_LENGTH = 4
MAX_PACKET_LENGTH = 64

CMD_MEASUREMENT = 0x95
CMD_PPG = 0x96
CMD_HRV = 0x97
CMD_BATTERY = 0x99
CMD_RAW = 0x85
CMD_UID = 0x1D
CMD_CONTROL = 0xB0

VERIFIED_COMMANDS = {CMD_RAW, CMD_PPG}


def u16_be(data: bytes) -> int:
    return (data[0] << 8) | data[1]


def u24_be(data: bytes) -> int:
    return (data[0] << 16) | (data[1] << 8) | data[2]


def checksum(length: int, device_id: int, command: int, payload: bytes) -> int:
    return (length + device_id + command + sum(payload)) & 0xFF


def build_packet(command: int, payload: bytes = b"", device_id: int = DEVICE_ID) -> bytes:
    length = 4 + len(payload)
    packet_checksum = checksum(length, device_id, command, payload)
    return HEADER + bytes([length, packet_checksum, device_id, command]) + payload


@dataclass
class Packet:
    raw: bytes
    length: int
    checksum: int
    device_id: int
    command: int
    payload: bytes
    valid_checksum: bool


def decode_measurement(payload: bytes) -> str:
    if len(payload) < 8:
        return f"measurement(payload={payload.hex(' ')})"

    flag_byte = payload[0]
    pulse_rate = payload[1] | ((payload[2] & 0x01) << 8)
    spo2 = (payload[2] >> 1) & 0x7F

    temperature = None
    if payload[3] not in (0x00, 0x7F):
        temperature = payload[3] + (payload[4] / 100.0)

    pi = None
    if payload[5] not in (0x00, 0x7F):
        pi = payload[5] + (payload[6] / 100.0)

    respiration_rate = payload[7] if payload[7] != 0 else None

    parts = [
        f"flags=0x{flag_byte:02X}",
        f"sync={(flag_byte >> 7) & 1}",
        f"no_pulse={(flag_byte >> 6) & 1}",
        f"searching={(flag_byte >> 5) & 1}",
        f"pulse_rate={pulse_rate}",
        f"spo2={spo2}",
    ]
    parts.append("temperature=NA" if temperature is None else f"temperature={temperature:.2f}")
    parts.append("pi=NA" if pi is None else f"pi={pi:.2f}")
    parts.append(f"resp_rate={respiration_rate if respiration_rate is not None else 'NA'}")
    return "measurement(" + ", ".join(parts) + ")"


def decode_ppg(payload: bytes) -> str:
    if len(payload) != 20:
        return f"ppg(payload={payload.hex(' ')})"

    samples = []
    for index in range(0, 20, 2):
        sample_0 = payload[index]
        sample_1 = payload[index + 1]
        samples.append(f"{sample_0 & 0x7F}/{sample_1 & 0x0F}")
    return "ppg(wave/bar=[" + ", ".join(samples) + "])"


def decode_hrv(payload: bytes) -> str:
    if len(payload) != 20:
        return f"hrv(payload={payload.hex(' ')})"

    intervals = [u16_be(payload[index : index + 2]) for index in range(0, 20, 2)]
    return "hrv(intervals_ms=[" + ", ".join(str(value) for value in intervals) + "])"


def decode_battery(payload: bytes) -> str:
    if not payload:
        return "battery(level=NA)"
    return f"battery(level={payload[0]})"


def decode_raw(payload: bytes) -> str:
    if len(payload) != 9:
        return f"raw(payload={payload.hex(' ')})"
    red = u24_be(payload[0:3])
    infrared = u24_be(payload[3:6])
    background = u24_be(payload[6:9])
    return f"raw(red={red}, ir={infrared}, bck={background})"


def decode_uid(payload: bytes) -> str:
    return f"uid(data={payload.hex(' ')})"


def command_provenance(command: int) -> str:
    return "verified" if command in VERIFIED_COMMANDS else "provisional"


def decode_packet(packet: Packet) -> str:
    if packet.command == CMD_MEASUREMENT:
        return decode_measurement(packet.payload)
    if packet.command == CMD_PPG:
        return decode_ppg(packet.payload)
    if packet.command == CMD_HRV:
        return decode_hrv(packet.payload)
    if packet.command == CMD_BATTERY:
        return decode_battery(packet.payload)
    if packet.command == CMD_RAW:
        return decode_raw(packet.payload)
    if packet.command == CMD_UID:
        return decode_uid(packet.payload)
    if packet.command == CMD_CONTROL:
        return f"control(payload={packet.payload.hex(' ')})"
    return f"cmd=0x{packet.command:02X}(payload={packet.payload.hex(' ')})"


def parse_packets(buffer: bytearray) -> Iterator[Packet]:
    while True:
        header_index = buffer.find(HEADER)
        if header_index < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            return
        if header_index > 0:
            del buffer[:header_index]
        if len(buffer) < 3:
            return

        length = buffer[2]
        if length < MIN_PACKET_LENGTH or length > MAX_PACKET_LENGTH:
            # Length byte is invalid; drop one byte and re-scan to recover from stream desync.
            del buffer[0]
            continue

        frame_size = length + 2
        if len(buffer) < frame_size:
            return

        raw = bytes(buffer[:frame_size])
        del buffer[:frame_size]

        if len(raw) < 6:
            continue

        packet_checksum = raw[3]
        device_id = raw[4]
        command = raw[5]
        payload = raw[6:]
        expected = checksum(length, device_id, command, payload)
        yield Packet(
            raw=raw,
            length=length,
            checksum=packet_checksum,
            device_id=device_id,
            command=command,
            payload=payload,
            valid_checksum=packet_checksum == expected,
        )


def format_packet(packet: Packet) -> str:
    status = "ok" if packet.valid_checksum else "bad-checksum"
    decoded = decode_packet(packet)
    return (
        f"[{status}|{command_provenance(packet.command)}] len={packet.length} dev=0x{packet.device_id:02X} "
        f"cmd=0x{packet.command:02X} cs=0x{packet.checksum:02X} {decoded}"
    )


def open_serial(port: str, baudrate: int) -> serial.Serial:
    return serial.Serial(port=port, baudrate=baudrate, timeout=0.1)


def read_packets(ser: serial.Serial, seconds: float) -> Iterator[Packet]:
    buffer = bytearray()
    deadline = None if seconds <= 0 else time.monotonic() + seconds
    while deadline is None or time.monotonic() < deadline:
        chunk = ser.read(256)
        if chunk:
            buffer.extend(chunk)
            yield from parse_packets(buffer)


def send_command(ser: serial.Serial, command: int, payload: bytes = b"") -> bytes:
    packet = build_packet(command, payload)
    ser.write(packet)
    ser.flush()
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description="AM801 protocol implementation")
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=230400)
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="Listen duration; use 0 to run until interrupted",
    )
    parser.add_argument("--start", action="store_true", help="Send the data transmission start command")
    parser.add_argument("--stop", action="store_true", help="Send the data transmission stop command")
    parser.add_argument("--query-uid", action="store_true", help="Send the UID query command")
    args = parser.parse_args()

    with open_serial(args.port, args.baud) as ser:
        print(f"Opened {args.port} @ {args.baud} baud")

        if args.stop:
            packet = send_command(ser, CMD_CONTROL, b"\x00")
            print(f"sent: {packet.hex(' ')}")
        if args.start:
            packet = send_command(ser, CMD_CONTROL, b"\x01")
            print(f"sent: {packet.hex(' ')}")
        if args.query_uid:
            packet = send_command(ser, CMD_UID)
            print(f"sent: {packet.hex(' ')}")

        try:
            for packet in read_packets(ser, args.seconds):
                print(format_packet(packet))
        except KeyboardInterrupt:
            print("Stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())