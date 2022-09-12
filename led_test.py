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
device.contrast(8 * 16)
draw.rounded_rectangle(device.bounding_box, outline="white")
time.sleep(0.5)
device.contrast(16 * 16)
draw.rounded_rectangle(device.bounding_box, outline="white")