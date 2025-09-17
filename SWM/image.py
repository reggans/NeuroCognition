from PIL import Image, ImageDraw, ImageFont

import random
import os

class SWMImage:
    def __init__(self, save_path, n_boxes, box_width=50, img_size=(800, 400), margin=30):
        self.margin = margin
        self.img_size = (img_size[0] + 2 * self.margin, img_size[1] + 2 * self.margin)
        self.base_img = Image.new('RGB', self.img_size, color = 'black')
        self.base_draw = ImageDraw.Draw(self.base_img)
        self.box_width = box_width
        self.n_boxes = n_boxes
        self.box_coords = []
        self.save_path = save_path
        self.padding = 5  # Padding to make boxes smaller than grid cells
        self.token_colors = ['red', 'blue', 'green', 'purple',]

        x_max, y_max = img_size
        font_size = 15
        try:
            self.font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            self.font = ImageFont.load_default()

        # Draw horizontal lines and y-coordinates
        for y in range(0, y_max + 1, self.box_width):
            self.base_draw.line([(self.margin, y + self.margin), (x_max + self.margin, y + self.margin)], fill='gray', width=1)
            if y < y_max:
                self.base_draw.text((self.margin // 3, y + self.box_width // 2 + self.margin - font_size // 2), str(y // self.box_width), fill='white', font=self.font)

        # Draw vertical lines and x-coordinates
        for x in range(0, x_max + 1, self.box_width):
            self.base_draw.line([(x + self.margin, self.margin), (x + self.margin, y_max + self.margin)], fill='gray', width=1)
            if x < x_max:
                 self.base_draw.text((x + self.box_width // 2 + self.margin - font_size // 2, self.margin // 3), str(x // self.box_width), fill='white', font=self.font)


        # Randomize box
        for i in range(self.n_boxes):
            while True:
                x = random.randint(0, x_max // self.box_width - 1)
                y = random.randint(0, y_max // self.box_width - 1)
                if (x, y) not in self.box_coords:
                    break
            self.box_coords.append((x, y))

        for coords in self.box_coords:
            x, y = self._convert_grid_to_coords(*coords)
            self._draw_box((x, y))
        
        self.base_img.save(os.path.join(self.save_path, 'base.png'))
        self.base_img.save(os.path.join(self.save_path, 'current.png'))

    def _draw_box(self, box_center):
        self.base_draw.rectangle((box_center[0] - self.box_width/2 + self.padding, box_center[1] - self.box_width/2 + self.padding,
                        box_center[0] + self.box_width/2 - self.padding, box_center[1] + self.box_width/2 - self.padding),
                        fill='yellow', outline='black', width=2)

    def _convert_grid_to_coords(self, grid_x, grid_y):
        x = grid_x * self.box_width + self.box_width / 2 + self.margin
        y = grid_y * self.box_width + self.box_width / 2 + self.margin
        return x, y

    def open_box(self, box_coord, token_box):
        if len(token_box) > len(self.token_colors):
            raise ValueError(f"Number of tokens {len(token_box)} exceeds available colors {len(self.token_colors)}")

        if box_coord not in self.box_coords:
            self.base_img.save(os.path.join(self.save_path, f'current.png'))
            return self.base_img
        
        box = self.get_box_id(box_coord)
        box_center = self._convert_grid_to_coords(*box_coord)
        new_img = self.base_img.copy()
        draw = ImageDraw.Draw(new_img)
        
        # Draw the opened box (black interior) with same padding as original boxes
        hole_padding = self.padding + 5
        draw.rectangle((box_center[0] - self.box_width/2 + hole_padding, box_center[1] - self.box_width/2 + hole_padding,
                            box_center[0] + self.box_width/2 - hole_padding, box_center[1] + self.box_width/2 - hole_padding),
                            fill='black')

        if box in token_box.values():
            # Get all instances of this box in the tokens list
            token_colors = [color for color, b in token_box.items() if b == box]
            num_tokens = len(token_colors)

            if num_tokens == 1:
                # Single token - fill the entire opened box
                token_color = token_colors[0]
                token_padding = self.padding + 10
                draw.rectangle((box_center[0] - self.box_width/2 + token_padding, box_center[1] - self.box_width/2 + token_padding,
                                box_center[0] + self.box_width/2 - token_padding, box_center[1] + self.box_width/2 - token_padding),
                                fill=token_color)
            else:
                # Multiple tokens - divide the box space
                token_padding = self.padding + 10
                box_inner_width = self.box_width - 2 * token_padding
                box_inner_height = self.box_width - 2 * token_padding
                
                if num_tokens == 2:
                    # Split horizontally for 2 tokens
                    token_width = box_inner_width // 2
                    for i, token_color in enumerate(token_colors):
                        x_start = box_center[0] - self.box_width/2 + token_padding + i * token_width
                        draw.rectangle((x_start, box_center[1] - self.box_width/2 + token_padding,
                                        x_start + token_width, box_center[1] + self.box_width/2 - token_padding),
                                        fill=token_color)
                elif num_tokens == 3:
                    # Split into 2 on top, 1 on bottom
                    token_height = box_inner_height // 2
                    token_width = box_inner_width // 2
                    
                    # Top two tokens
                    for i in range(2):
                        token_color = token_colors[i]
                        x_start = box_center[0] - self.box_width/2 + token_padding + i * token_width
                        draw.rectangle((x_start, box_center[1] - self.box_width/2 + token_padding,
                                        x_start + token_width, box_center[1] - self.box_width/2 + token_padding + token_height),
                                        fill=token_color)
                    
                    # Bottom token (centered)
                    token_color = token_colors[2]
                    draw.rectangle((box_center[0] - token_width/2, box_center[1] - self.box_width/2 + token_padding + token_height,
                                    box_center[0] + token_width/2, box_center[1] + self.box_width/2 - token_padding),
                                    fill=token_color)
                else:
                    # 4 or more tokens - split into 2x2 grid
                    token_width = box_inner_width // 2
                    token_height = box_inner_height // 2

                    for i, token_color in enumerate(token_colors[:4]):  # Limit to 4 tokens max
                        row = i // 2
                        col = i % 2
                        x_start = box_center[0] - self.box_width/2 + token_padding + col * token_width
                        y_start = box_center[1] - self.box_width/2 + token_padding + row * token_height
                        draw.rectangle((x_start, y_start,
                                        x_start + token_width, y_start + token_height),
                                        fill=token_color)
        
        new_img.save(os.path.join(self.save_path, f'current.png'))
        
        return new_img
    
    def get_box_id(self, box_coord):
        return self.box_coords.index(box_coord) + 1  # Box IDs are 1-indexed
    
    def get_box_coord(self, box_id):
        return self.box_coords[box_id - 1]  # Convert to 0-indexed for list access