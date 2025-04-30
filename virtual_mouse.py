import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading
import speech_recognition as sr

# Constants
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
exit_flag = False

# Capture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

plocX, plocY = 0, 0
clocX, clocY = 0, 0
screenW, screenH = pyautogui.size()
prevY = 0
clickCooldown = 1
lastClickTime = 0

def get_finger_state(lmList):
    finger_tips = [4, 8, 12, 16, 20]
    state = []
    for tip in finger_tips:
        if tip == 4:
            state.append(lmList[tip][1] < lmList[tip - 1][1])
        else:
            state.append(lmList[tip][2] < lmList[tip - 2][2])
    return state

# 🔊 Voice recognition thread
def voice_listener():
    global exit_flag
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    while not exit_flag:
        try:
            with mic as source:
                print("Listening for 'close mouse'...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()
                print(f"Voice Command: {command}")
                if "close mouse" in command:
                    print("Voice command received: Closing virtual mouse.")
                    exit_flag = True
        except:
            continue

# Start voice thread
voice_thread = threading.Thread(target=voice_listener, daemon=True)
voice_thread.start()

# 🖱️ Main loop
while not exit_flag:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        lmList = []

        for id, lm in enumerate(handLms.landmark):
            cx, cy = int(lm.x * wCam), int(lm.y * hCam)
            lmList.append((id, cx, cy))
            cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)

        if lmList:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            x3, y3 = lmList[16][1:]
            fingers = get_finger_state(lmList)
            currentTime = time.time()

            if fingers[1] and not fingers[2]:
                x3_screen = np.interp(x1, (frameR, wCam - frameR), (0, screenW))
                y3_screen = np.interp(y1, (frameR, hCam - frameR), (0, screenH))
                clocX = plocX + (x3_screen - plocX) / smoothening
                clocY = plocY + (y3_screen - plocY) / smoothening
                pyautogui.moveTo(screenW - clocX, clocY)
                plocX, plocY = clocX, clocY
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)

                if prevY != 0:
                    deltaY = y1 - prevY
                    if deltaY > 20:
                        pyautogui.scroll(-30)
                    elif deltaY < -20:
                        pyautogui.scroll(30)
                prevY = y1
            else:
                prevY = 0

            dist_left_click = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            if fingers[1] and fingers[2] and dist_left_click < 40:
                if currentTime - lastClickTime > clickCooldown:
                    pyautogui.click()
                    lastClickTime = currentTime
                    cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)

            dist_right_click = ((x2 - x3)**2 + (y2 - y3)**2)**0.5
            if fingers[2] and fingers[3] and dist_right_click < 40:
                if currentTime - lastClickTime > clickCooldown:
                    pyautogui.click(button='right')
                    lastClickTime = currentTime
                    cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

exit_flag = True
cap.release()
cv2.destroyAllWindows()
