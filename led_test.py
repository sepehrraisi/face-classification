import re
import time
import argparse

from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.virtual import viewport
from luma.core.legacy import text, show_message
from luma.core.legacy.font import proportional, CP437_FONT, TINY_FONT, SINCLAIR_FONT, LCD_FONT

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=2, block_orientation=-90,
                     rotate=0, blocks_arranged_in_reverse_order=False)

with canvas(device) as draw:
    draw.rectangle(device.bounding_box, outline="white")
    text(draw, (2, 2), "Hello", fill="white", font=proportional(LCD_FONT))
    text(draw, (2, 10), "World", fill="white", font=proportional(LCD_FONT))
    time.sleep(2)


with canvas(device) as draw:
    draw.arc(device.bounding_box, 0, 128, fill="white")

for _ in range(5):
    for intensity in range(16):
        device.contrast(intensity * 16)
        time.sleep(0.1)

with canvas(device) as draw:
    draw.bitmap(device.bounding_box)

for _ in range(5):
    for intensity in range(16):
        device.contrast(intensity * 16)
        time.sleep(0.1)
with canvas(device) as draw:
    draw.chord(device.bounding_box)

for _ in range(5):
    for intensity in range(16):
        device.contrast(intensity * 16)
        time.sleep(0.1)

with canvas(device) as draw:
    draw.ellipse(device.bounding_box)

for _ in range(5):
    for intensity in range(16):
        device.contrast(intensity * 16)
        time.sleep(0.1)
with canvas(device) as draw:
    draw.line(device.bounding_box)

for _ in range(5):
    for intensity in range(16):
        device.contrast(intensity * 16)
        time.sleep(0.1)

with canvas(device) as draw:
    draw.shape(device.bounding_box, outline="white")

for _ in range(5):
    for intensity in range(16):
        device.contrast(intensity * 16)
        time.sleep(0.1)

with canvas(device) as draw:
    draw.pieslice(device.bounding_box, outline="white")

for _ in range(5):
    for intensity in range(16):
        device.contrast(intensity * 16)
        time.sleep(0.1)