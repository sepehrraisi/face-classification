import time
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=2, block_orientation=-90,
                 rotate=0, blocks_arranged_in_reverse_order=False)

def normal():
    with canvas(device) as draw:
        draw.rectangle((0, 3, 15, 4), fill="white", width=1)
    for _ in range(1):
        for intensity in range(8):
            device.contrast(intensity * 16)
            time.sleep(0.1)


def sad():
    with canvas(device) as draw:
        draw.arc((0, 0, 15, 7), -10, 200, fill="white", width=2)

    for _ in range(1):
        for intensity in range(8):
            device.contrast(intensity * 16)
            time.sleep(0.1)

def smile():
    with canvas(device) as draw:
        draw.arc((0, 2, 15, 9), 170, 9, fill="white", width=2)
        print(device.bounding_box)

    for _ in range(1):
        for intensity in range(8):
            device.contrast(intensity * 16)
            time.sleep(0.1)