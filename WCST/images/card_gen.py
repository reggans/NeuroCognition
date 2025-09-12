"""
From https://github.com/OttoKuosmanen/WCST
"""

from PIL import Image, ImageOps

star = 'shapes/star.png'
circle = 'shapes/circle.png'
square = 'shapes/square.png'
triangle = 'shapes/triangle.png'
frame = 'shapes/frame.png'

star = Image.open(star)
circle = Image.open(circle)
square = Image.open(square)
triangle = Image.open(triangle)
frame = Image.open(frame)

# Convert frame to RGBA
frame = frame.convert('RGBA')

# Create a white background image
white_bg = Image.new('RGBA', frame.size, (255, 255, 255, 255))
print(frame.size)

# Paste the frame onto the white background
frame = Image.alpha_composite(white_bg, frame)

red = (255, 36, 0)
green = (11, 218, 81)
blue = (25, 25, 112)
yellow = (254, 219, 0)

# Background colors (lighter versions)
bg_red = (255, 182, 182)
bg_green = (182, 255, 182)
bg_blue = (182, 182, 255)
bg_yellow = (255, 255, 182)

shapes = [circle, square, triangle, star]
colors = [blue, yellow, red, green]
bg_colors = [bg_red, bg_green, bg_blue, bg_yellow]  # Corresponding background colors
numbers = [1, 2, 3, 4]

positions = {
    1: [(46, 70)],
    2: [(25, 70), (67, 70)],
    3: [(25, 80), (67, 80), (46, 42)],
    4: [(25, 92), (67, 92), (25, 48), (67, 48)]
}

def create_cards(numbers, shapes, colors, bg_colors, positions):
    for i, shape in enumerate(shapes):
        for j, color in enumerate(colors):
            for number in numbers:
                black, transparent = shape.split()
                changeling = ImageOps.colorize(black, color, color)
                changeling.putalpha(transparent)
                
                # Deduce shape and color names for saving
                s = ["circle", "square", "triangle", "star"][shapes.index(shape)]
                c = ["blue", "yellow", "red", "green"][colors.index(color)]
                bg_name = ["red", "green", "blue", "yellow"][i]  # Background color based on shape index
                
                # Create card with background color
                card = frame.copy()
                
                # Apply background color (from the predefined mapping)
                bg_color = bg_colors[i]  # Background color corresponds to shape
                bg_overlay = Image.new('RGBA', card.size, bg_color + (200,))  # Semi-transparent
                card = Image.alpha_composite(card, bg_overlay)
                
                # Add shapes
                for pos in positions[number]:
                    card.paste(changeling, pos, mask=changeling)
                
                card.save(f"cards/{number}_{s}_{c}_{bg_name}_bg.png")

create_cards(numbers, shapes, colors, bg_colors, positions)