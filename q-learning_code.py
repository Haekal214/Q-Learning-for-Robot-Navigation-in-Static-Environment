import cv2
import numpy as np
import random
import time
import csv
import os
from matplotlib import pyplot as plt
from epuck import epuck

def cluster_lines(lines, rho_thresh=50, theta_thresh=np.deg2rad(2)):
    clustered = []
    for r_theta in lines:
        rho, theta = r_theta[0]
        added = False
        for group in clustered:
            avg_rho = np.mean([l[0] for l in group])
            avg_theta = np.mean([l[1] for l in group])
            if abs(rho - avg_rho) < rho_thresh and abs(theta - avg_theta) < theta_thresh:
                group.append((rho, theta))
                added = True
                break
        if not added:
            clustered.append([(rho, theta)])  
    return [np.mean(group, axis=0) for group in clustered]

def merge_close_lines(lines_with_midpoint, x_thresh=20, y_thresh=20):
    merged = []
    used = [False] * len(lines_with_midpoint)
    for i in range(len(lines_with_midpoint)):
        if used[i]:
            continue
        ((rho_i, theta_i), (x1_i, y1_i, x2_i, y2_i), (mid_x_i, mid_y_i)) = lines_with_midpoint[i]
        is_vertical = abs(x1_i - x2_i) < abs(y1_i - y2_i)
        group = [lines_with_midpoint[i]]
        used[i] = True

        for j in range(i+1, len(lines_with_midpoint)):
            if used[j]:
                continue
            ((rho_j, theta_j), (x1_j, y1_j, x2_j, y2_j), (mid_x_j, mid_y_j)) = lines_with_midpoint[j]
            if is_vertical:
                if abs(mid_x_i - mid_x_j) < x_thresh:
                    group.append(lines_with_midpoint[j])
                    used[j] = True
            else:
                if abs(mid_y_i - mid_y_j) < y_thresh:
                    group.append(lines_with_midpoint[j])
                    used[j] = True
        avg_x1 = int(np.mean([g[1][0] for g in group]))
        avg_y1 = int(np.mean([g[1][1] for g in group]))
        avg_x2 = int(np.mean([g[1][2] for g in group]))
        avg_y2 = int(np.mean([g[1][3] for g in group]))
        avg_mid_x = int(np.mean([g[2][0] for g in group]))
        avg_mid_y = int(np.mean([g[2][1] for g in group]))
        avg_rho = np.mean([g[0][0] for g in group])
        avg_theta = np.mean([g[0][1] for g in group])
        merged.append(((avg_rho, avg_theta), (avg_x1, avg_y1, avg_x2, avg_y2), (avg_mid_x, avg_mid_y)))
    return merged

def check_finish_points(occupancy, pixel_coordinates, img, radius, x_coord, is_vertical=False):
    finish_coords = []
    temp_finish_coords = []
    consecutive_free_count = 0
    for y in range(1, 7):  
        if is_vertical:
            coord = (y, x_coord)  
        else:
            coord = (x_coord, y)  
        if occupancy.get(coord, False) is False:
            consecutive_free_count += 1
            temp_finish_coords.append(coord)
            if consecutive_free_count == 6:
                for i in range(y - consecutive_free_count + 1, y + 1):
                    if is_vertical:
                        occupancy[(i, x_coord)] = True  
                        mid_x, mid_y = pixel_coordinates[(i, x_coord)]  
                    else:
                        occupancy[(x_coord, i)] = True  
                        mid_x, mid_y = pixel_coordinates[(x_coord, i)]  
                    cv2.circle(img, (mid_x, mid_y), radius, (0, 0, 255), -1)  
                temp_finish_coords.clear()
            else:
                mid_x, mid_y = pixel_coordinates[coord]  
                cv2.circle(img, (mid_x, mid_y), radius, (255, 0, 0), -1)  
        else:
            consecutive_free_count = 0
    finish_coords.extend(temp_finish_coords)
    return finish_coords

def is_yellow_dominant(image, x, y, radius=10, threshold=0.5):
    y_min = max(y - radius, 0)
    y_max = min(y + radius, image.shape[0])
    x_min = max(x - radius, 0)
    x_max = min(x + radius, image.shape[1])
    cropped_area = image[y_min:y_max, x_min:x_max]
    hsv_area = cv2.cvtColor(cropped_area, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_area, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    total_pixels = cropped_area.size // 3  

    if total_pixels > 0:
        yellow_ratio = yellow_pixels / total_pixels
        return yellow_ratio > threshold
    return False

def is_robot_present(image, x, y, radius=30, threshold=0.007):
    y_min = max(y - radius, 0)
    y_max = min(y + radius, image.shape[0])
    x_min = max(x - radius, 0)  
    x_max = min(x + radius, image.shape[1])
    cropped_area = image[y_min:y_max, x_min:x_max]
    hsv_area = cv2.cvtColor(cropped_area, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv_area, lower_green, upper_green)
    green_pixels = cv2.countNonZero(green_mask)
    total_pixels = cropped_area.size // 3  
    
    if total_pixels > 0:
        green_ratio = green_pixels / total_pixels
        return green_ratio > threshold
    return False

original_image = cv2.imread("C:/Users/bhara/OneDrive - Officeku/Documents/Skripsi/ArenaIRL/arena2.jpg")
if original_image is None:
    raise FileNotFoundError("Gambar tidak ditemukan.")

height, width = original_image.shape[:2]
if height > width:  
    original_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)

original_image = cv2.resize(original_image, (1600, 902))
blurred = cv2.GaussianBlur(original_image, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(red_mask1, red_mask2)
kernel = np.ones((5, 5), np.uint8)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

x_min, y_min, x_max, y_max = original_image.shape[1], original_image.shape[0], 0, 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    x_min = min(x_min, x)
    y_min = min(y_min, y)
    x_max = max(x_max, x + w)
    y_max = max(y_max, y + h)

margin = 10
x_min = max(x_min - margin, 0)
y_min = max(y_min - margin, 0)
x_max = min(x_max + margin, original_image.shape[1])
y_max = min(y_max + margin, original_image.shape[0])
cropped_image = original_image[y_min:y_max, x_min:x_max]
cv2.imwrite("C:/Users/bhara/OneDrive - Officeku/Documents/Skripsi/ArenaIRL/arena_crop.jpg", cropped_image)
img = cropped_image.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 70, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

if lines is not None:
    merged_lines = cluster_lines(lines)
    lines_with_midpoint = []
    for (rho, theta) in merged_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        if abs(x1 - x2) < abs(y1 - y2):  
            if y1 < y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
        else:  
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1

        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        lines_with_midpoint.append(((rho, theta), (x1, y1, x2, y2), (mid_x, mid_y)))

    vertical_lines = [line for line in lines_with_midpoint if abs(line[1][0] - line[1][2]) < abs(line[1][1] - line[1][3])]
    horizontal_lines = [line for line in lines_with_midpoint if abs(line[1][0] - line[1][2]) >= abs(line[1][1] - line[1][3])]
    vertical_lines_sorted = sorted(vertical_lines, key=lambda x: x[2][0])
    horizontal_lines_sorted = sorted(horizontal_lines, key=lambda x: x[2][1])
    all_lines_sorted = vertical_lines_sorted + horizontal_lines_sorted
    final_vertical = merge_close_lines(vertical_lines_sorted)
    final_horizontal = merge_close_lines(horizontal_lines_sorted)
    
    for idx, (rhotheta, (x1, y1, x2, y2), (mid_x, mid_y)) in enumerate(final_vertical):
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    for idx, (rhotheta, (x1, y1, x2, y2), (mid_x, mid_y)) in enumerate(final_horizontal):
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    x_first = [line[1][0] for line in final_vertical]
    x_last = [line[2][0] for line in final_vertical]
    y_first = [line[1][1] for line in final_horizontal]
    y_last = [line[2][1] for line in final_horizontal]
    x_coords = []
    y_coords = []
    for i in range(len(x_first)):
        if x_first[i] < x_last[i]:
            increment = (x_last[i] - x_first[i]) / 7
            x_coords.append([x_first[i] + increment * j for j in range(8)])
        else:
            decrement = (x_first[i] - x_last[i]) / 7
            x_coords.append([x_first[i] - decrement * j for j in range(8)])
    for i in range(len(y_first)):
        if y_first[i] < y_last[i]:
            increment = (y_last[i] - y_first[i]) / 7
            y_coords.append([y_first[i] + increment * j for j in range(8)])
        else:
            decrement = (y_first[i] - y_last[i]) / 7
            y_coords.append([y_first[i] - decrement * j for j in range(8)])
    
    grid_width = len(x_coords) - 1
    grid_height = len(y_coords) - 1
    occupancy = {(xi, yi): False for xi in range(grid_width) for yi in range(grid_height)}
    pixel_coordinates = {}
    robot_coordinates = []
    finish_coords = []
    temp_finish_coords = []
    
    for xi in range(grid_width):
        for yi in range(grid_height):
            x_label = xi
            y_label = yi
            x_first = x_coords[xi][0 + yi]
            x_last = x_coords[xi + 1][0 + yi]
            y_first = y_coords[-(yi + 1)][0 + xi]
            y_last = y_coords[-(yi + 2)][0 + xi]
            mid_x = int((x_first + x_last) / 2)
            mid_y = int((y_first + y_last) / 2)
            pixel_coordinates[(x_label, y_label)] = (mid_x, mid_y)

            if is_robot_present(cropped_image, mid_x, mid_y):
                robot_coordinates.append((x_label, y_label))

            color = (0, 255, 0)  
            radius = 5
            is_yellow = is_yellow_dominant(cropped_image, mid_x, mid_y)
            occupancy[(x_label, y_label)] = is_yellow  

            if x_label == 0 or x_label == 7 or y_label == 0 or y_label == 7:
                obstacles_on_x0 = [occupancy.get((0, yl), False) for yl in range(grid_height)]
                obstacles_on_y0 = [occupancy.get((xl, 0), False) for xl in range(grid_width)]
                obstacles_count_x0 = sum(obstacles_on_x0)
                obstacles_count_y0 = sum(obstacles_on_y0)
                if occupancy[(x_label, y_label)]:
                    color = (0, 255, 255)  
                else:
                    color = (0, 0, 255)  
                    if (x_label == 0 and y_label == 0) or (x_label == 7 and y_label == 0) or \
                    (x_label == 0 and y_label == 7) or (x_label == 7 and y_label == 7):
                        occupancy[(x_label, y_label)] = True  
            else:
                if occupancy[(x_label, y_label)]:
                    color = (0, 255, 255)  
                else:
                    color = (0, 255, 0)  
            cv2.circle(img, (mid_x, mid_y), radius, color, -1)
            cv2.putText(img, f"({x_label},{y_label})", (mid_x + 10, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    finish_coords = []   
    finish_coords.extend(check_finish_points(occupancy, pixel_coordinates, img, radius, 7))
    finish_coords.extend(check_finish_points(occupancy, pixel_coordinates, img, radius, 0))
    finish_coords.extend(check_finish_points(occupancy, pixel_coordinates, img, radius, 0, is_vertical=True))
    finish_coords.extend(check_finish_points(occupancy, pixel_coordinates, img, radius, 7, is_vertical=True))
else:
    print("Tidak ditemukan.")

cv2.imwrite("C:/Users/bhara/OneDrive - Officeku/Documents/Skripsi/ArenaIRL/linesDetected_filtered.jpg", img)
all_coords = list(pixel_coordinates.keys())
sorted_coords = sorted(all_coords, key=lambda coord: (coord[0], coord[1]))

print("\n--- Hasil Deteksi ---")
print("Koordinat:")
for (x_label, y_label) in sorted_coords:
    print(f"({x_label}, {y_label})")

print("\nKoordinat yang dapat dilalui:")
walkable_coords = [(x, y) for (x, y), is_obstacle in occupancy.items() if not is_obstacle]
sorted_walkable_coords = sorted(walkable_coords, key=lambda coord: (coord[0], coord[1]))
for (x, y) in sorted_walkable_coords:
    print(f"({x}, {y})")

print("\nKoordinat yang tidak dapat dilalui:")
blocked_coords = [(x, y) for (x, y), is_obstacle in occupancy.items() if is_obstacle]
sorted_blocked_coords = sorted(blocked_coords, key=lambda coord: (coord[0], coord[1]))
for (x, y) in sorted_blocked_coords:
    print(f"({x}, {y})")

print("\nKoordinat robot:")
for (x, y) in robot_coordinates:
    print(f"({x}, {y})")

print("\nKoordinat finish:")
sorted_finish_coords = sorted(finish_coords, key=lambda coord: (coord[0], coord[1]))
for (x, y) in sorted_finish_coords:
    print(f"({x}, {y})")

print("Hasil disimpan sebagai linesDetected_filtered.jpg")
rows, cols = 6, 6
aksi = ['timur', 'barat', 'utara', 'selatan']
peta_aksi = {
    'timur': (1, 0),
    'barat': (-1, 0),
    'utara': (0, 1),
    'selatan': (0, -1)
}
q_table = {}
epk = epuck(port='COM4',debug=False)
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 500
folder_tujuan = r'C:\Users\bhara\OneDrive - Officeku\Documents\Skripsi\Code Project\Skripsi\Webots\Data Q-Table2'
forward_speed = 100
rotate_speed = -100
tile_size = 0.25
prev_float_x, prev_float_y, after_xpos, after_ypos, prev_gridx, prev_gridy = None, None, None, None, None, None
pause = False
pause_start_time = 0.0
current_direction = 0  
step_count, check_step = 0, 1
movement_condition = True


def get_state(posisi):
    return f"{posisi[0]},{posisi[1]}"

def posisi_valid(posisi):
    return (posisi not in sorted_blocked_coords)

def lakukan_aksi(posisi, aksi, finish):
    gerak = peta_aksi[aksi]
    posisi_baru = (posisi[0] + gerak[0], posisi[1] + gerak[1])
    if posisi_baru in finish:
        return posisi_baru, 100  
    elif posisi_valid(posisi_baru):
        return posisi_baru, -1   
    else:
        return posisi, -5        

def cari_nama_file():
    os.makedirs(folder_tujuan, exist_ok=True)  
    idx = 1
    while os.path.exists(os.path.join(folder_tujuan, f'hasil_labirin{idx}.csv')):
        idx += 1
    return os.path.join(folder_tujuan, f'hasil_labirin{idx}.csv')

for r in range(1, rows+1):
    for c in range(1, cols+1):
        q_table[get_state((r, c))] = {a: 0.0 for a in aksi}
sorted_finish_coords = sorted(finish_coords, key=lambda coord: (coord[0], coord[1]))
for (x, y) in sorted_finish_coords:
    q_table[get_state((x, y))] = {a: 0.0 for a in aksi}  

hasil_data = []

for ep in range(1, episodes + 1):
    for (x, y) in robot_coordinates:
        posisi = (x, y)
    langkah = 0
    total_reward = 0
    jalur_aksi = []
    q_sums = {get_state((r, c)): {a: 0.0 for a in aksi} for r in range(1, rows+1) for c in range(1, cols+1)}
    goal_tercapai = False

    while posisi not in sorted_finish_coords and langkah < 150:
        state = get_state(posisi)
        if random.random() < epsilon:
            aksi_dipilih = random.choice(aksi)
        else:
            max_q = max(q_table[state].values())
            max_actions = [a for a, q in q_table[state].items() if q == max_q]
            aksi_dipilih = random.choice(max_actions)

        posisi_baru, reward = lakukan_aksi(posisi, aksi_dipilih, sorted_finish_coords)
        next_state = get_state(posisi_baru)
        aksi_terbaik_berikut = max(q_table[next_state], key=q_table[next_state].get)
        delta_q = alpha * (reward + gamma * q_table[next_state][aksi_terbaik_berikut] - q_table[state][aksi_dipilih])
        q_table[state][aksi_dipilih] += delta_q
        
        if state in q_sums:
            q_sums[state][aksi_dipilih] += delta_q
    
        jalur_aksi.append(aksi_dipilih)
        posisi = posisi_baru
        total_reward += reward
        langkah += 1

        if posisi in sorted_finish_coords:
            goal_tercapai = True
            break
    
    for r in range(1, rows+1):
        for c in range(1, cols+1):
            state = get_state((r, c))
            record = {
                'Episode': ep,
                'State': state,
                'Utara': q_table[state]['utara'],
                'Timur': q_table[state]['timur'],
                'Selatan': q_table[state]['selatan'],
                'Barat': q_table[state]['barat'],
                'Goal': goal_tercapai,
                'Total Aksi': langkah,
                'Reward': total_reward
            }
            hasil_data.append(record)   
    
    hasil_data.append({
        'Episode': ep,
        'State': 'Jalur',
        'Utara': '',
        'Timur': '',
        'Selatan': '',
        'Barat': '',
        'Goal': '',
        'Total Aksi': '',
        'Reward': ','.join(jalur_aksi)
    })

nama_file = cari_nama_file()
with open(nama_file, mode='w', newline='') as file:
    fieldnames = ['Episode', 'State', 'Utara', 'Timur', 'Selatan', 'Barat', 'Goal', 'Total Aksi', 'Reward']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for row in hasil_data:
        writer.writerow(row)

print(f"Data hasil pelatihan disimpan di: {nama_file}")

def baca_data_csv(nama_file):
    data = []
    with open(nama_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def cari_jalur_terbaik(data):
    jalur = []
    for (x, y) in robot_coordinates:
        posisi = (x, y)
    state = f"{posisi[0]},{posisi[1]}"
    
    episode_terakhir = max(int(row['Episode']) for row in data)
    
    while state not in sorted_finish_coords:  
        state_data = [row for row in data if row['Episode'] == str(episode_terakhir) and row['State'] == state]
        if state_data:
            q_values = {
                'utara': float(state_data[0]['Utara']),
                'timur': float(state_data[0]['Timur']),
                'selatan': float(state_data[0]['Selatan']),
                'barat': float(state_data[0]['Barat'])
            }
            aksi_terbaik = max(q_values, key=q_values.get)
            jalur.append(aksi_terbaik)
            gerak = peta_aksi[aksi_terbaik]
            posisi = (posisi[0] + gerak[0], posisi[1] + gerak[1])
            state = f"{posisi[0]},{posisi[1]}"
        else:
            break  
    return jalur

def konversi_jalur_ke_angka(jalur):
    arah_ke_angka = {
        "timur": 0,
        "selatan": 1,
        "barat": 2,
        "utara": 3
    }
    return [arah_ke_angka[arah] for arah in jalur]

data = baca_data_csv(nama_file)
jalur_terbaik = cari_jalur_terbaik(data)

print("\nJalur terbaik yang harus ditempuh:")
print(" -> ".join(jalur_terbaik))
jalur_terbaik_angka = konversi_jalur_ke_angka(jalur_terbaik)
time_forward = 1.96
time_turn = 0.647
time_turnaround = 1.294
motor_maju = 500
motor_mundur = -500
motor_diam = 0
pause = 1
current_direction = 0  

def wait_exact(duration):
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        pass

def rotate_robot(target_direction):
    global current_direction
    rotation = (target_direction - current_direction) % 4
    if rotation == 1:
        print(f"Rotating right 90°, duration: {time_turn} s")
        epk.setMotorSpeed(motor_maju, motor_mundur)
        wait_exact(time_turn)
    elif rotation == 3:
        print(f"Rotating left 90°, duration: {time_turn} s")
        epk.setMotorSpeed(motor_mundur, motor_maju)
        wait_exact(time_turn)
    elif rotation == 2:
        print(f"Turning around 180°, duration: {time_turnaround} s")
        epk.setMotorSpeed(motor_mundur, motor_maju)
        wait_exact(time_turnaround)
    else:
        print("No rotation, moving forward")
        epk.setMotorSpeed(motor_maju, motor_maju)
        wait_exact(time_forward)
        epk.setMotorSpeed(motor_diam, motor_diam)
        return 

    epk.setMotorSpeed(motor_diam, motor_diam)
    current_direction = target_direction

def movement_robot(direction):
    epk.setMotorSpeed(motor_diam, motor_diam)
    wait_exact(pause)

    rotate_robot(direction)

    if current_direction == direction:
        epk.setMotorSpeed(motor_maju, motor_maju)
        wait_exact(time_forward)
        epk.setMotorSpeed(motor_diam, motor_diam)

    wait_exact(pause)
    return direction

for arah in jalur_terbaik_angka:  
    current_direction = movement_robot(arah)