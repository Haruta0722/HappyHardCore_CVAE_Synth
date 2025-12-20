# this is a code for jetson to test Bluetooth receiving character "Hello world"

from bluezero import peripheral

SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"

def write_callback(value, options):
    text = bytes(value).decode("utf-8")
    print("Received:", text)

periph = peripheral.Peripheral(
    adapter_addr=None,
    local_name="JetsonBLE",
    appearance=0x0000
)

periph.add_service(
    srv_id=1,
    uuid=SERVICE_UUID,
    primary=True
)

periph.add_characteristic(
    srv_id=1,
    chr_id=1,
    uuid=CHAR_UUID,
    value=[],
    notifying=False,
    flags=["write"],
    write_callback=write_callback
)

periph.publish()

print("BLE Peripheral running...")
periph.run()
