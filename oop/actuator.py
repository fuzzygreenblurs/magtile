import serial
import time
from constants import *

class Actuator:
    _driven_coils = []

    @classmethod
    def set_driven_coils(cls, driven):
        cls._driven_coils = driven

    @classmethod
    def get_driven_coils(cls):
        return cls._driven_coils

    def __init__(self, port, baudrate=115200, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(1)  # Wait for Arduino to reset
        self._clear_initial_message()  # Clear the "Command Line Terminal Ready" message
        self.scan_addresses()  # Perform initial scan

    def _clear_initial_message(self):
        # Read and discard the initial "Command Line Terminal Ready" message
        initial_message = self.ser.readline().decode().strip()
        if initial_message != "Command Line Terminal Ready":
            raise ValueError(f"Unexpected initial message: {initial_message}")
    
    def _send_command(self, command):
        self.ser.write(f"{command}\n".encode())
        response = self.ser.readline().decode().strip()
        if response.startswith("ok : "):
            return response[5:]
        elif response == "error":
            raise ValueError(f"Error executing command: {command}")
        else:
            raise ValueError(f"Unexpected response: {response}")
        
    def actuate(self, row, col, dc=4000):
        self.set_power(row, col, dc)
        print(f"Stopped all coils first. Now setting power for coil_id: ({row}, {col}) to {round((dc/4095) * 100, 2)}%")
        
    def actuate_single(self, row, col, dc=4000):
        self.set_driven_coils(self.get_driven_coils.append([row, col]))
        self.set_power(row, col, dc)
        time.sleep(0.3)
        print(f"Setting power for coil_id: ({row}, {col}) to {round((dc/4095) * 100, 2)}%")
        # print(f"Stopped all coils first. Now setting power for coil_id: ({row}, {col}) to {round((dc/4095) * 100, 2)}%")

    def stop_all_driven(self):
        for coil in self.get_driven_coils():
            self.set_power(coil[0], coil[1], 0)
        self.set_driven_coils = []

    def stop_all(self):
        # print(f"stopping all coils...")
        for row in range(16):
            for col in range(16):
                self.set_power(row, col, 0)

    def read_width(self):
        return int(self._send_command("read_width"))

    def read_height(self):
        return int(self._send_command("read_height"))

    def write_width(self, width):
        self._send_command(f"write_width {width}")

    def write_height(self, height):
        self._send_command(f"write_height {height}")

    def write_address_list(self, addresses):
        address_str = " ".join(map(str, addresses))
        self._send_command(f"write_address_list {address_str}")

    def read_address_list(self):
        return list(map(int, self._send_command("read_address_list").split()))

    def scan_addresses(self):
        return list(map(int, self._send_command("scan_addresses").split()))

    def test_led_enable(self, address):
        self._send_command(f"test_led_enable {address}")

    def test_led_disable(self, address):
        self._send_command(f"test_led_disable {address}")

    def store_config(self):
        self._send_command("store_config")

    def set_power(self, row, col, power):
        self._send_command(f"set_power {row} {col} {power}")

    def get_power(self, row, col):
        return int(self._send_command(f"get_power {row} {col}"))

    def blinkall_start(self):
        self._send_command("blinkall_start")

    def blinkall_stop(self):
        self._send_command("blinkall_stop")

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()