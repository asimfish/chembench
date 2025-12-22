#!/usr/bin/env python3
import argparse
import sys
from . import __version__
from .serial_interface import SerialInterface
from .psi_glove_controller import PSIGloveController


def device_info():
    parser = argparse.ArgumentParser(description="PSI Glove device info")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port path")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baudrate")
    args = parser.parse_args()

    serial = SerialInterface(args.port, args.baudrate, auto_connect=True, mock=False)
    ctrl = PSIGloveController(serial)
    ok = ctrl.is_connected()
    print(f"psi-glove-sdk v{__version__}")
    print(f"Connected: {ok}")
    if not ok:
        print("Hint: check permissions (Linux: sudo usermod -a -G dialout $USER)")
        sys.exit(1)
    status = ctrl.loop()
    if status:
        print("Thumb joints:", status.thumb)
    ctrl.disconnect()

if __name__ == "__main__":
    device_info()
EOF
