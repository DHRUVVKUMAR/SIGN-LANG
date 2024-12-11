import cv2

cap = cv2.VideoCapture(0)  # Try 0, 1, or 2 if multiple cameras exist
if not cap.isOpened():
    print("Error: Camera not accessible.")
else:
    print("Camera is working!")

    while True:
        ret, frame = cap.read()  # Read a frame
        if not ret:
            print("Failed to capture frame")
            break

        # Display the captured frame
        cv2.imshow('Camera Feed', frame)

        # Wait for the user to press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
