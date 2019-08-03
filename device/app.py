from device import Device
import sys

app = Device(__name__, sys.argv[1])

if __name__ == '__main__':
    app.run()