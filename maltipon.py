import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load images
imgBackground = cv2.imread("Resources/Background.png")
imgGameOver = cv2.imread("Resources/gameOver.png")
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Game variables
frameCount = 0
speedI = 200
ballPos = [100, 100]
speedX = 15
speedY = 15
gameOver = False
score = [0, 0]

def get_closest_right_hand(face_center, hands):
    """Find the right hand closest to the given face center."""
    min_dist = float('inf')
    closest_hand = None

    for hand in hands:
        if hand.get('type', '').lower() != 'right':
            continue

        x, y, w, h = hand['bbox']
        hand_center = (x + w // 2, y + h // 2)
        dist = np.linalg.norm(np.array(face_center) - np.array(hand_center))
        
        if dist < min_dist:
            min_dist = dist
            closest_hand = hand

    return closest_hand


while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # Face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Show face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)

    # Show number of faces
    cv2.putText(img, f'Faces: {len(faces)}', (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    # Hand detection
    hands, img = detector.findHands(img, flipType=False)

    # Overlay background
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    # Game logic: only allow if 2 faces and 2 hands
    if len(faces) != 2 or len(hands) != 2:
        gameOver = True

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[0] + score[1]).zfill(2), (585, 360),
                    cv2.FONT_HERSHEY_COMPLEX, 2.5, (200, 0, 200), 5)
    else:
        # Sort faces left to right
        faces = sorted(faces, key=lambda f: f[0])
        left_face_center = (faces[0][0] + faces[0][2] // 2, faces[0][1] + faces[0][3] // 2)
        right_face_center = (faces[1][0] + faces[1][2] // 2, faces[1][1] + faces[1][3] // 2)

        # Filter right hands only
        right_hands = [hand for hand in hands if hand['type'] == "Right"]

        if len(right_hands) == 2:
            # Sort right hands by x-coordinate (left to right)
            right_hands.sort(key=lambda h: h['bbox'][0])
            left_hand = right_hands[0]
            right_hand = right_hands[1]
        else:
            left_hand = None
            right_hand = None


        # Draw and control left bat
        if left_hand:
            x, y, w, h = left_hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)
            img = cvzone.overlayPNG(img, imgBat1, (59, y1))
            if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                speedX = -speedX
                ballPos[0] += 30
                score[0] += 1

        # Draw and control right bat
        if right_hand:
            x, y, w, h = right_hand['bbox']
            h2, w2, _ = imgBat2.shape
            y2 = y - h2 // 2
            y2 = np.clip(y2, 20, 415)
            img = cvzone.overlayPNG(img, imgBat2, (1195, y2))
            if 1195 - 50 < ballPos[0] < 1195 and y2 < ballPos[1] < y2 + h2:
                speedX = -speedX
                ballPos[0] -= 30
                score[1] += 1

        # Speed up over time
        frameCount += 1
        if frameCount % speedI == 0:
            speedX += 10 if speedX > 0 else -10
            speedY += 10 if speedY > 0 else -10

        # Ball boundary logic
        if ballPos[0] < 40 or ballPos[0] > 1200:
            gameOver = True
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        # Move ball
        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw ball and score
        img = cvzone.overlayPNG(img, imgBall, ballPos)
        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX,
                    3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX,
                    3, (255, 255, 255), 5)

    # Small preview
    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    # Display
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # Reset game
    if key == ord('r'):
        frameCount = 0
        ballPos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]
