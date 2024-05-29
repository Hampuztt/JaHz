import sys

print("This is a message from python", flush=True)

while True:
    received_string = sys.stdin.readline().strip()
    if received_string:
        print(received_string, flush=True)

