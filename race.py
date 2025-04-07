import cv2
import numpy as np
import time
import math
import serial
from picamera2 import Picamera2

raw_size = (3200, 2400)
crop_params = (20, 1208, 3200, 1200)  # (x, y, w, h)
output_size = (800, 600)

Kp = 0.4
Ki = 0.0
Kd = 0.0
prev_error = 0
integral = 0

motor_speed = 80      # مقدار سرعت موتور (0 تا 255)
move_command = 'F' 

# اتصال به آردوینو
try:
    arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    time.sleep(2)  # صبر برای راه‌اندازی کامل آردوینو
except Exception as e:
    print(f"Error: Could not connect to Arduino: {e}")
    exit()
    

def send_command(command):
    try:
        command += '\n'  # اضافه کردن کاراکتر newline برای تکمیل دستور
        arduino.write(command.encode())
        print(f"Sent: {command.strip()}")
    except Exception as e:
        print(f"Error: Failed to send command: {e}")

    # ارسال دستور تنظیم سرعت موتور
send_command(str(motor_speed))
time.sleep(2)
    # ارسال فرمان حرکت (مثلاً حرکت به جلو)
send_command(move_command)
time.sleep(2)
picam2 = Picamera2()
config = picam2.create_preview_configuration(
main={"size": output_size, "format": "RGB888"},
raw={"size": raw_size},
controls={"ScalerCrop": crop_params, "FrameRate": 10})
picam2.configure(config)
picam2.start()
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window", 640, 480 )   

cv2.namedWindow("line detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("line detection", 640, 480)
#cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Original Video", 640, 480)

src_points = np.array([[0, 2400], [3200, 2400], [3200, 1000], [0, 1000]], dtype=np.float32)
width, height = 920, 700
dst_points = np.array([[0, height], [width, height], [width, 0], [0, 0]], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

def find_white_mean_x(image):
    # Load the image
    # image = cv2.imread(image)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define yellow color range in HSV
    #lower_white = np.array([0, 0, 50])   # حداقل مقدار برای خاکستری (روشن‌تر)
    #upper_white = np.array([255, 50, 200])
    
    # Define red color range in HSV
    lower_red = np.array([0, 0, 200])   # حداقل مقدار برای خاکستری (روشن‌تر)
    upper_red = np.array([180, 40, 255])

    
    # Threshold the image to get only yellow colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Get the x-coordinates of yellow pixels
    red_pixels = np.column_stack(np.where(mask > 0))  # (y, x) format
    
    if red_pixels.size == 0:
        #print("No yellow-like color detected.")
        return 1000
    
    mean_x = np.mean(red_pixels[:, 1])  # Extract x-coordinates and find mean
    # print(f"Mean x-coordinate of yellow pixels: {mean_x}")
    return mean_x

def reduce_local_brightness(image, threshold=1, gamma=1.0, kernel_size=10):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    _, bright_mask = cv2.threshold(v, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    v_norm = v / 255.0
    v_gamma = np.power(v_norm, gamma)
    v_gamma = np.uint8(v_gamma * 255)
    v_adjusted = v.copy()
    v_adjusted[bright_mask == 255] = v_gamma[bright_mask == 255]
    hsv_adjusted = cv2.merge([h, s, v_adjusted])
    return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

def pid_control(error):
    print(f'error: {error}')
    global prev_error, integral
    integral += error
    derivative = error - prev_error
    prev_error = error
    control = Kp * error + Ki * integral + Kd * derivative
    return control

    import numpy as np

def compute_slope(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return float('inf')  # vertical line
    return (y2 - y1) / (x2 - x1)

def find_the_4_lines_with_closest_slope(lines):
    lines_array = lines.reshape(-1, 4)
    n = len(lines_array)
    if n < 4:
        print("Need at least 4 lines")

    # Calculate slope for each line
    slopes = np.array([
        compute_slope(x1, y1, x2, y2)
        for x1, y1, x2, y2 in lines_array
    ])

    # Sort lines by slope
    sorted_indices = np.argsort(slopes)
    sorted_lines = lines_array[sorted_indices]
    sorted_slopes = slopes[sorted_indices]

    # Now find 4 consecutive lines with minimum max slope difference
    min_diff = float('inf')
    best_group = None

    for i in range(n - 3):
        group = sorted_lines[i:i+4]
        group_slopes = sorted_slopes[i:i+4]
        diff = np.max(group_slopes) - np.min(group_slopes)
        if diff < min_diff:
            min_diff = diff
            best_group = group

    return best_group


def Line_Detection(img):
    k=False
    
    height, width = img.shape[:2]
    
    
    mask = np.zeros_like(img)
    polygon = np.array([[ (50, 500), (750,500), (360 , 50), (360, 50 )]], np.int32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian Blur to reduce noise
    mask = np.zeros_like(blur) 
    cv2.fillPoly(mask, [polygon], (255, 255, 255))
  # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.bitwise_and(edges, mask)  # Apply the mask to the image
    cv2.polylines(img, [polygon], isClosed=True, color=(0, 0, 0), thickness=2)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=100)

    left_lines, right_lines = [], []
    if lines is not None:
        
        #lines = find_the_4_lines_with_closest_slope(lines)
        #lines_array = lines.reshape(-1, 1, 4)
        #print(lines)
        #print("-------------------------------")
        
        for line in lines:
            
            x1, y1, x2, y2 = line[0]
            slope=abs((y2-y1)//(x2-x1))
            #arz = x2 - x1
            if x1 < width // 2 and x2 < width // 2 and (slope>2):
                left_lines.append(line)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif x1 > width // 2 and x2 > width // 2 and (slope>5):
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                right_lines.append(line)

    def average_line(lines):
        
        if not lines:
            return None
        x1_avg = sum(line[0][0] for line in lines) // len(lines)
        y1_avg = sum(line[0][1] for line in lines) // len(lines)
        x2_avg = sum(line[0][2] for line in lines) // len(lines)
        y2_avg = sum(line[0][3] for line in lines) // len(lines)
        return [[x1_avg, y1_avg, x2_avg, y2_avg]]

    avg_left_line = average_line(left_lines)
    avg_right_line = average_line(right_lines)

    if avg_left_line:
        x1, y1, x2, y2 = avg_left_line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 30)

    if avg_right_line:
        x1, y1, x2, y2 = avg_right_line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 30)

    # نقطه میانی مسیر
    if avg_left_line and avg_right_line:
        left_center = ((avg_left_line[0][0] + avg_left_line[0][2]) // 2, (avg_left_line[0][1] + avg_left_line[0][3]) // 2)
        right_center = ((avg_right_line[0][0] + avg_right_line[0][2]) // 2, (avg_right_line[0][1] + avg_right_line[0][3]) // 2)
        center_point = ((left_center[0] + right_center[0]) // 2, (left_center[1] + right_center[1]) // 2)

    elif avg_right_line:
        #center_point = ((avg_right_line[0][0] + avg_right_line[0][2]) // 2, (avg_right_line[0][1] + avg_right_line[0][3]) // 2)
        center_point = None
        k=True
    else:
        center_point = None#-----------------------------------------------------
    
    if k :
            servo_angle=10
            print("LK")
            k=False
            return servo_angle

            
    if center_point :
        cv2.circle(img, center_point, 5, (0, 255, 0), 100)
        
        car_center_x = width // 2
        #print(f"car center:{car_center_x}")
        car_center_x =car_center_x
        l=center_point[0]
        #print(f"center_point : {center_point[0]}")
        error = (l - car_center_x)
        
        #print(f"error: {error}")
        
        steering_angle = int(np.clip(pid_control(error), -100, 100))
        print(f'Steering Angle: {steering_angle}')
        steering_angle = ((steering_angle*1.1)+110)
        if steering_angle==0:
           steering_angle = 50
        else:
            steering_angle=steering_angle/2
            #if find_white_mean_x 
            print(find_white_mean_x(img))
            servo_angle = int(steering_angle)

        #print(f'Steering Angle: {servo_angle}')
        return servo_angle


while True:
    image = picam2.capture_array()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    image = cv2.rotate(image, cv2.ROTATE_180)
    image = reduce_local_brightness(image, threshold=1, gamma=1.5, kernel_size=10)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    frame = image
    #frame = cv2.warpPerspective(image, matrix, (width, height))
    for point in src_points:
      cv2.circle(image, tuple(point.astype(int)), 5, (0, 0, 255), -1)
    cv2.polylines(image, [src_points.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
    #frame = reduce_local_brightness(warped) 

    #Line_Detection(frame)  
    servo_angle = 45  # مقدار پیش‌فرض در صورت عدم تشخیص خط
    temp_angle = Line_Detection(frame)
    if temp_angle is not None:
      servo_angle = temp_angle
      send_command(f"A{servo_angle}")
    cv2.imshow('line detection', frame)  
    #cv2.imshow('Original Video', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        
        move_command = 'S'
        send_command(move_command)
        time.sleep(2) 
        cv2.destroyAllWindows()
        break

