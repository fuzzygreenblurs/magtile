import serial
import time
from tile_controller import TileController

def activate(controller, row1, col1, row2, col2, dc1=4095, dc2=4095, sleep_time=1):    
    controller.set_power(row1, col1, dc1)
    controller.set_power(row2, col2, dc2)
    print(f"Set power for coils on tile ({row1}, {col1}) to {(dc1/4095) * 100}%, and ({row2}, {col2}) to {(dc2/4095) * 100}%")
    time.sleep(sleep_time)
    controller.set_power(row1, col1, 0)
    controller.set_power(row2, col2, 0)

def activate_single(controller, row1, col1, dc1=4095, sleep_time=3):    
    controller.set_power(row1, col1, dc1)
    print(f"Set power for coils on tile ({row1}, {col1}) to {round((dc1/4095) * 100, 2)}%")
    time.sleep(0.3)
    controller.set_power(row1, col1, 0)
    time.sleep(sleep_time)

def activate_pair(controller, pair, dc=4000, sleep_time=1):
    for coil in pair:
        controller.set_power(coil[0], coil[1], dc)
        print(f"Set power for coils on tile ({coil[0]}, {coil[1]}) to {round((dc/4095) * 100, 2)}%")

    time.sleep(sleep_time)

    for coil in pair:
        controller.set_power(coil[0], coil[1], 0)
        print(f"Set power for coils on tile ({coil[0]}, {coil[1]}) to {round((dc/4095) * 100, 2)}%")


def move_magnet_loop(controller, pair1, pair2, pair3, pair4, num_cycles=20):
    for i in range(num_cycles):
        activate_pair(controller, pair1)
        activate_pair(controller, pair2)
        activate_pair(controller, pair3)
        activate_pair(controller, pair4)
        print(f"Cycle {i+1}/{num_cycles} completed")

def main():
    port = "/dev/cu.usbmodem21301"  # Update with the correct port for your setup
    try:
        with TileController(port) as controller:
            try:
                # Read dimensions
                width = controller.read_width()
                height = controller.read_height()
                print(f"Width: {width}, Height: {height}")

                addresses = controller.scan_addresses()
                print(f"Addresses: {addresses}")

                # Store configuration
                controller.store_config()

                try:
                    pair1 = [[11, 3], [11, 11]]
                    pair2 = [[11, 4], [11, 12]]
                    pair3 = [[10, 4], [10, 12]]
                    pair4 = [[10, 3], [10, 11]]

                    move_magnet_loop(controller, pair1, pair2, pair3, pair4)
                except KeyboardInterrupt:
                    stop_all(controller)

            except Exception as e:
                print(f"An error occurred during operations: {e}")
    except serial.SerialException:
        print("Failed to connect to the Arduino.")

if __name__ == "__main__":
    main()
