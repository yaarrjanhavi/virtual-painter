import cv2
import mediapipe as mp
import numpy as np

# ---------- GUI helpers ----------

def nothing(x):
    pass

# Color picker window (RGB trackbars)
cv2.namedWindow("Color Picker")
cv2.resizeWindow("Color Picker", 300, 150)
cv2.createTrackbar("R", "Color Picker", 255, 255, nothing)
cv2.createTrackbar("G", "Color Picker", 255, 255, nothing)
cv2.createTrackbar("B", "Color Picker", 0,   255, nothing)

# Brush size window
cv2.namedWindow("Brush")
cv2.resizeWindow("Brush", 300, 80)
cv2.createTrackbar("Size", "Brush", 8, 50, nothing)

# ---------- Camera & Mediapipe ----------

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,          # 1 = better accuracy than 0 [web:53]
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ---------- Drawing state ----------

canvas = None
prev_x, prev_y = 0, 0

# Colors (BGR); last one is "CUSTOM" from trackbars
colors = [
    (255, 0, 255),   # Purple
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 255, 255),   # Yellow
    (0, 0, 0)        # Custom placeholder
]
color_names = ["PURPLE", "BLUE", "GREEN", "YELLOW", "CUSTOM"]
current_color = colors[0]


def fingers_up(hand):
    """Return [index_up, middle_up] using landmark y positions."""
    fingers = []
    fingers.append(hand.landmark[8].y < hand.landmark[6].y)    # index
    fingers.append(hand.landmark[12].y < hand.landmark[10].y)  # middle
    return fingers


def draw_palette(img):
    """Draw color palette and CLEAR button at the top."""
    h, w, _ = img.shape

    # read custom color from trackbars
    r = cv2.getTrackbarPos("R", "Color Picker")
    g = cv2.getTrackbarPos("G", "Color Picker")
    b = cv2.getTrackbarPos("B", "Color Picker")
    custom_color = (b, g, r)     # BGR
    colors[-1] = custom_color    # update CUSTOM swatch

    box_w = w // (len(colors) + 1)  # +1 for CLEAR

    # draw color boxes
    for i, col in enumerate(colors):
        x1 = i * box_w
        x2 = (i + 1) * box_w
        cv2.rectangle(img, (x1, 0), (x2, 60), col, -1)
        cv2.putText(img, color_names[i], (x1 + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # draw CLEAR button
    x1 = len(colors) * box_w
    x2 = (len(colors) + 1) * box_w
    cv2.rectangle(img, (x1, 0), (x2, 60), (50, 50, 50), -1)
    cv2.putText(img, "CLEAR", (x1 + 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    alpha = 1.3   # contrast
    beta = 20     # brightness
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    draw_palette(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    mode = "NONE"

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            index_up, middle_up = fingers_up(hand)

            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            if index_up and middle_up:
                # -------- SELECT mode --------
                mode = "SELECT"
                prev_x, prev_y = 0, 0

                if y < 60:
                    box_w = w // (len(colors) + 1)
                    idx = x // box_w

                    if idx < len(colors):
                        # select palette color (includes CUSTOM)
                        current_color = colors[idx]
                    else:
                        # tapped CLEAR
                        canvas = np.zeros((h, w, 3), dtype=np.uint8)

                cv2.circle(frame, (x, y), 15, current_color, cv2.FILLED)

            elif index_up and not middle_up:
                # -------- DRAW / ERASE mode --------
                mode = "DRAW"
                cv2.circle(frame, (x, y), 10, current_color, cv2.FILLED)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                # brush size from trackbar
                size = cv2.getTrackbarPos("Size", "Brush")
                if size <= 0:
                    size = 1

                if current_color == (0, 0, 0):   # eraser
                    thickness = max(40, size * 3)
                else:
                    thickness = size

                cv2.line(canvas, (prev_x, prev_y), (x, y),
                         current_color, thickness)
                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = 0, 0
    else:
        prev_x, prev_y = 0, 0

    # simple blending of canvas and webcam frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 1.0, 0)

    cv2.putText(frame, f"Mode: {mode}",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    cv2.putText(
        frame,
        "Index: Draw | Index+Middle: Select/Clear | C: Clear | S: Save | Q: Quit",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1
    )

    cv2.imshow("Air Canvas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if key == ord('s'):
        cv2.imwrite("drawing.png", canvas)
        print("Saved drawing.png")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
