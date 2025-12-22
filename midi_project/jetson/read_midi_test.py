import os

fd = os.open("/dev/snd/midiC1D0", os.O_RDONLY)

while True:
    data = os.read(fd, 3)
    print(data[1])
