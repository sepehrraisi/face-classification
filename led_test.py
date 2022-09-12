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

# with canvas(device) as draw:
#     draw.arc((0, 0, 15, 7), -10, 200, fill="white", width=2)
#
# for _ in range(2):
#     for intensity in range(16):
#         device.contrast(intensity * 16)
#         time.sleep(0.1)
#
# with canvas(device) as draw:
#     draw.arc((0, 2, 15, 9), 170, 9, fill="white", width=2)
#     print(device.bounding_box)
#
# for _ in range(2):
#     for intensity in range(16):
#         device.contrast(intensity * 16)
#         time.sleep(0.1)

with canvas(device) as draw:
    draw.line((4, 0, 15, 2), fill="white", width=2)
for _ in range(2):
    for intensity in range(16):
        device.contrast(intensity * 16)
        time.sleep(0.1)
