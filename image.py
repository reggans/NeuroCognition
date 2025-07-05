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
        for i in range(num_boxes):
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

    def _draw_box(self, box_center):
        self.base_draw.rectangle((box_center[0] - self.box_width/2, box_center[1] - self.box_width/2,
                        box_center[0] + self.box_width/2, box_center[1] + self.box_width/2),
                        fill='yellow')

    def _convert_grid_to_coords(self, grid_x, grid_y):
        x = grid_x * self.box_width + self.box_width / 2 + self.margin
        y = grid_y * self.box_width + self.box_width / 2 + self.margin
        return x, y

    def open_box(self, box_coord, token):
        if box_coord not in self.box_coords:
            return self.base_img
        
        box = self.box_coords.index(box_coord)
        box_center = self._convert_grid_to_coords(*box_coord)
        new_img = self.base_img.copy()
        draw = ImageDraw.Draw(new_img)
        
        draw.rectangle((box_center[0] - self.box_width/2 + 5, box_center[1] - self.box_width/2 + 5,
                            box_center[0] + self.box_width/2 - 5, box_center[1] + self.box_width/2 - 5),
                            fill='black')

        if token == box:
            draw.rectangle((box_center[0] - self.box_width/2 + 15, box_center[1] - self.box_width/2 + 15,
                            box_center[0] + self.box_width/2 - 15, box_center[1] + self.box_width/2 - 15),
                            fill='red')
        
        new_img.save(os.path.join(self.save_path, f'current.png'))
        
        return new_img