from PIL import Image, ImageOps, ImageDraw, ImageFont
import os


def draw_five_cards(
    given_card_attributes,
    option_card_attributes=None,
    bg_color=False,
    save_path="WCST/images/current.png",
):
    """
    Draw 5 cards on a white board with specified attributes.

    Args:
        given_card_attributes (dict): Dictionary with keys 'shape', 'color', 'count'
                                    shape: 'circle', 'triangle', 'star', 'square'
                                    color: 'red', 'green', 'blue', 'yellow'
                                    count: integer (1-4)
    """
    # Load shape images
    shapes_dir = "WCST/images/shapes/"
    star_img = Image.open(shapes_dir + "star.png")
    circle_img = Image.open(shapes_dir + "circle.png")
    square_img = Image.open(shapes_dir + "square.png")
    triangle_img = Image.open(shapes_dir + "triangle.png")
    frame_img = Image.open(shapes_dir + "frame.png")

    # Convert frame to RGBA and create white background
    frame_img = frame_img.convert("RGBA")
    white_bg = Image.new("RGBA", frame_img.size, (255, 255, 255, 255))
    frame_img = Image.alpha_composite(white_bg, frame_img)

    # Define colors
    red = (255, 36, 0)
    green = (11, 218, 81)
    blue = (25, 25, 112)
    yellow = (254, 219, 0)

    # Shape and color mappings
    shape_map = {
        "circle": circle_img,
        "square": square_img,
        "triangle": triangle_img,
        "star": star_img,
    }

    color_map = {"red": red, "green": green, "blue": blue, "yellow": yellow}

    # Define positions for shapes on cards (from card_gen.py)
    positions = {
        1: [(46, 70)],
        2: [(25, 70), (67, 70)],
        3: [(25, 80), (67, 80), (46, 42)],
        4: [(25, 92), (67, 92), (25, 48), (67, 48)],
    }

    # Define the first 4 cards attributes.
    # Backward-compatible fallback: canonical 4 cards when options are not provided.
    if option_card_attributes is not None:
        if len(option_card_attributes) != 4:
            raise ValueError("option_card_attributes must contain exactly 4 card dicts")
        predefined_cards = option_card_attributes
    elif bg_color:
        predefined_cards = [
            {"shape": "circle", "color": "red", "count": 1, "background": "red"},
            {"shape": "triangle", "color": "green", "count": 2, "background": "green"},
            {"shape": "star", "color": "blue", "count": 3, "background": "blue"},
            {"shape": "square", "color": "yellow", "count": 4, "background": "yellow"},
        ]
    else:
        predefined_cards = [
            {"shape": "circle", "color": "red", "count": 1},
            {"shape": "triangle", "color": "green", "count": 2},
            {"shape": "star", "color": "blue", "count": 3},
            {"shape": "square", "color": "yellow", "count": 4},
        ]

    # Create white board (larger canvas for 5 cards)
    board_width = 800
    board_height = 300
    board = Image.new("RGBA", (board_width, board_height), (255, 255, 255, 255))

    # Card dimensions and positions
    card_width, card_height = frame_img.size
    margin = 20
    spacing = 140
    extra_spacing = 50  # Extra space before the "Given" card

    # Calculate card positions
    start_x = margin
    card_positions = []
    for i in range(4):
        card_positions.append((start_x + i * spacing, margin))
    # Add extra space for the 5th card
    card_positions.append((start_x + 4 * spacing + extra_spacing, margin))

    # Create all 5 cards
    cards = []
    labels = ["1", "2", "3", "4", "Given"]

    for i in range(5):
        # Determine card attributes
        if i < 4:
            card_attrs = predefined_cards[i]
        else:
            card_attrs = given_card_attributes

        # Create card
        card = create_card(
            frame_img.copy(), card_attrs, shape_map, color_map, positions
        )
        cards.append(card)

    # Paste cards onto board and add labels
    draw = ImageDraw.Draw(board)

    for i, (card, (x, y)) in enumerate(zip(cards, card_positions)):
        board.paste(card, (x, y), card)

        # Add label below card
        label_x = x + card_width // 2
        label_y = y + card_height + 10

        try:
            # Try to use a larger font
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fall back to default font
            font = ImageFont.load_default()

        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), labels[i], font=font)
        text_width = bbox[2] - bbox[0]

        draw.text(
            (label_x - text_width // 2, label_y), labels[i], fill=(0, 0, 0), font=font
        )

    # Save the board image.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    board.save(save_path)
    # print(f"Cards saved to {save_path}")

    return board


def create_card(frame, attributes, shape_map, color_map, positions):
    """
    Create a single card with specified attributes using the same method as card_gen.py

    Args:
        frame: PIL Image of the card frame
        attributes: dict with 'shape', 'color', 'count', 'background'
        shape_map: mapping of shape names to PIL Images
        color_map: mapping of color names to RGB tuples
        positions: dict mapping count to list of (x, y) positions
    """
    shape_name = attributes["shape"]
    color_name = attributes["color"]
    count = attributes["count"]
    background_color = attributes.get("background", "white")

    # Get the shape image and color
    shape_img = shape_map[shape_name]
    color = color_map[color_name]

    # Create colored version of the shape (similar to card_gen.py)
    black, transparent = shape_img.split()
    colored_shape = ImageOps.colorize(black, color, color)
    colored_shape.putalpha(transparent)

    # Create the card with background color
    card = frame.copy()

    # Apply background color if not white
    if background_color != "white":
        # Define background color map - lighter versions of shape colors
        bg_color_map = {
            "red": (255, 182, 182),  # Light red
            "green": (182, 255, 182),  # Light green
            "blue": (182, 182, 255),  # Light blue
            "yellow": (255, 255, 182),  # Light yellow
            "white": (255, 255, 255),
        }

        bg_color = bg_color_map.get(background_color, (255, 255, 255))

        # Create a background overlay
        bg_overlay = Image.new("RGBA", card.size, bg_color + (200,))  # Semi-transparent
        card = Image.alpha_composite(card, bg_overlay)

    # Place shapes at specified positions
    for pos in positions[count]:
        card.paste(colored_shape, pos, mask=colored_shape)

    return card


# Example usage function
def show_cards(given_card_attributes):
    """
    Create and save the five cards to images/current.png.

    Args:
        given_card_attributes: dict with 'shape', 'color', 'count' for the 5th card
    """
    board = draw_five_cards(given_card_attributes)
    return board


if __name__ == "__main__":
    # Example: Draw cards with a given card having 2 red stars
    given_attributes = {"shape": "star", "color": "red", "count": 2}

    # Create and save the cards
    show_cards(given_attributes)
