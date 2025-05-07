import numpy as np
import pygame
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: disables oneDNN if you really want to avoid it

import tensorflow as tf

# Colors (modern pastel theme)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)
PASTEL_BLUE = (119, 191, 240)
PASTEL_GREEN = (119, 240, 119)
RED = (255, 0, 0)

# Constants
ROWS, COLS = 28, 28
CELL_SIZE = 12
ROBOTO = "assets/fonts/Roboto-Regular.ttf"
MODEL_PATH = "train/mnist_cnn_models.keras"  # Default model path


class DigitRecognizer:
    """
    Class for handling the digit recognition process using a neural network.

    Attributes
    ----------
    model : tf.keras.Model
        The trained model for digit classification.
    window_size : int
        The size of the pygame window.
    screen : pygame.Surface
        The surface to render the game.
    smallFont : pygame.font.Font
        Font for smaller text.
    largeFont : pygame.font.Font
        Font for larger text.
    handwriting : list of list of floats
        A 2D array representing the pixel values of the handwritten digit.
    classification : int or None
        The predicted classification of the digit.

    Methods
    -------
    run()
        Starts the game loop, handling events and rendering.
    handle_events()
        Handles events in the pygame window.
    draw_axis()
        Draws x and y axes on the screen.
    draw_grid_bounding_box(x, y)
        Draws the bounding box for the grid.
    draw()
        Renders the grid, buttons, and classification result.
    draw_button(x, y, label, callback)
        Draws a button and executes a callback when clicked.
    reset()
        Resets the grid and classification.
    classify()
        Classifies the handwritten input using the model.
    """

    def __init__(self, model_path):
        """
        Initializes the DigitRecognizer instance.

        Parameters
        ----------
        model_path : str
            Path to the pre-trained model file.
        """
        # Load model
        self.model = tf.keras.models.load_model(model_path,compile=False)

        # Pygame setup
        pygame.init()
        self.window_size = 500  # Square window size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Digit Recognizer")

        # Fonts
        self.smallFont = pygame.font.Font(ROBOTO, 20)
        self.largeFont = pygame.font.Font(ROBOTO, 40)

        # State
        self.handwriting = [[0] * COLS for _ in range(ROWS)]
        self.classification = None

    def run(self):
        """
        Starts the main loop for handling events and drawing on the screen.
        """
        while True:
            self.handle_events()
            self.draw()
            pygame.display.flip()

    def handle_events(self):
        """
        Handles user inputs and events, such as quitting the game.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def draw_axis(self):
        """
        Draws the x and y axes along with the origin point.
        """
        pygame.draw.line(self.screen, RED, (0, self.window_size // 2), (self.window_size, self.window_size // 2), 2)
        pygame.draw.line(self.screen, RED, (self.window_size // 2, 0), (self.window_size // 2, self.window_size), 2)
        pygame.draw.circle(self.screen, RED, (self.window_size // 2, self.window_size // 2), 5)

    def draw_grid_bounding_box(self, x, y):
        """
        Draws a bounding box for the grid, extending across the entire screen.

        Parameters
        ----------
        x : int
            The x-coordinate for the starting point of the grid.
        y : int
            The y-coordinate for the starting point of the grid.
        """
        pygame.draw.line(self.screen, RED, (x, 0), (x, self.window_size), 2)  # Left side
        pygame.draw.line(self.screen, RED, (x + CELL_SIZE * COLS, 0), (x + CELL_SIZE * COLS, self.window_size), 2)  # Right side
        pygame.draw.line(self.screen, RED, (0, y), (self.window_size, y), 2)  # Top side
        pygame.draw.line(self.screen, RED, (0, y + CELL_SIZE * ROWS), (self.window_size, y + CELL_SIZE * ROWS), 2)  # Bottom side

    def draw(self):
        """
        Renders the grid, buttons, and classification result on the screen.
        """
        self.screen.fill(DARK_GRAY)

        # Calculate the grid width and height
        grid_width = CELL_SIZE * COLS
        grid_height = CELL_SIZE * ROWS

        # Center the grid inside the window (without offset)
        grid_x = (self.window_size - grid_width) / 2
        grid_y = (self.window_size - grid_height) / 2

        mouse = pygame.mouse.get_pos() if pygame.mouse.get_pressed()[0] else None

        # Draw grid and handle drawing
        for i in range(ROWS):
            for j in range(COLS):
                rect = pygame.Rect(
                    grid_x + j * CELL_SIZE,
                    grid_y + i * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )

                # Fill cell if it has value (white writing on black background)
                if self.handwriting[i][j]:
                    val = int(self.handwriting[i][j] * 255)
                    pygame.draw.rect(self.screen, (val, val, val), rect)
                else:
                    pygame.draw.rect(self.screen, BLACK, rect)

                # pygame.draw.rect(self.screen, LIGHT_GRAY, rect, 0.1)  # Soft grid lines

                # Paint on cell
                if mouse and rect.collidepoint(mouse):
                    self.handwriting[i][j] = 250 / 255
                    if i + 1 < ROWS:
                        self.handwriting[i + 1][j] = 220 / 255
                    if j + 1 < COLS:
                        self.handwriting[i][j + 1] = 220 / 255
                    if i + 1 < ROWS and j + 1 < COLS:
                        self.handwriting[i + 1][j + 1] = 190 / 255

        # Calculate the space between the grid and the buttons
        button_y = grid_y + grid_height + 20  # 20px space between the grid and buttons

        # Draw buttons
        self.draw_button(grid_x, button_y, "Reset", self.reset)  # Reset button at the left
        self.draw_button(grid_x + grid_width - 120, button_y, "Classify", self.classify)  # Classify button at the right

        # Draw classification result
        if self.classification is not None:
            classificationText = self.largeFont.render(str(self.classification), True, PASTEL_BLUE)
            classificationRect = classificationText.get_rect()
            classificationRect.center = (self.window_size // 2, grid_y - 30)  # Position relative to the top of the grid
            self.screen.blit(classificationText, classificationRect)

    def draw_button(self, x, y, label, callback):
        """
        Draws a button and executes the provided callback when clicked.

        Parameters
        ----------
        x : int
            The x-coordinate for the button.
        y : int
            The y-coordinate for the button.
        label : str
            The text label for the button.
        callback : function
            The function to be called when the button is clicked.
        """
        button = pygame.Rect(
            x, y,  # Position of the button
            120, 40  # Size of the button
        )
        text = self.smallFont.render(label, True, WHITE)
        textRect = text.get_rect()
        textRect.center = button.center
        pygame.draw.rect(self.screen, PASTEL_GREEN, button)
        pygame.draw.rect(self.screen, BLACK, button, 3)  # Border for the button
        self.screen.blit(text, textRect)

        if pygame.mouse.get_pressed()[0]:
            mouse = pygame.mouse.get_pos()
            if button.collidepoint(mouse):
                callback()

    def reset(self):
        """
        Resets the grid and classification result.
        """
        self.handwriting = [[0] * COLS for _ in range(ROWS)]
        self.classification = None

    def classify(self):
        """
        Classifies the handwritten input using the trained model.

        Uses the current state of `handwriting`, reshapes it, and makes a prediction.

        Updates `classification` with the predicted class.
        """
        input_data = np.array(self.handwriting).reshape(1, 28, 28, 1)
        predictions = self.model.predict(input_data)
        self.classification = predictions.argmax()


def main():
    """
    Main function to run the digit recognition application.

    Checks for the provided model path argument, initializes the DigitRecognizer, and starts the application.
    """

    global MODEL_PATH
    if len(sys.argv) == 2:
        MODEL_PATH = sys.argv[1]  # Get model path from command line argument

    recognizer = DigitRecognizer(MODEL_PATH)
    recognizer.run()


if __name__ == "__main__":
    main()
