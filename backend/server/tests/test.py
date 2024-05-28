# receiver.py
import sys

if __name__ == "__main__":
    received_string = sys.stdin.read().strip()
    print(f"Received from Node.js: {received_string}")
    response_string = "Hello from Python"
    print(response_string)
