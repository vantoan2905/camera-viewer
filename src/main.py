import cv2
import pygame
import sys

def init_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None
    return cap

def main():
    """
    Camera Viewer App (Pygame + OpenCV)

    - Display images from the computer's camera.
    - Exit the application by pressing the 'q' key or closing the window.
    """
    cap = init_camera()
    if cap is None:
        return

    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Camera Viewer")

    clock = pygame.time.Clock()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (screen_width, screen_height))
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                cap.release()
                pygame.quit()
                sys.exit()

        clock.tick(30)

if __name__ == "__main__":
    main()
