
import cv2
import numpy as np
import time
import math
import serial
from picamera2 import Picamera2




motor_speed = 90 
 

# اتصال به آردوینو
try:
    arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    time.sleep(2)  # صبر برای راه‌اندازی کامل آردوینو
except Exception as e:
    #print(f"Error: Could not connect to Arduino: {e}")
    exit()
    

def send_command(command):
    try:
        command += '\n'  # اضافه کردن کاراکتر newline برای تکمیل دستور
        arduino.write(command.encode())
    except Exception as e:
        print(f"Error: Failed to send command: {e}")


send_command(str(motor_speed))
time.sleep(0.1)
send_command(f"A{50}")
time.sleep(0.1) 
send_command('F')
time.sleep(3.5)
send_command(f"A{10}")
time.sleep(0.1) 
send_command('F')
time.sleep(3.4)
send_command('S')
send_command(f"A{45}")
time.sleep(0.1)
print("end")
