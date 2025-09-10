
import os
import socket
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

def write_prompt_to_file(prompt: str) -> None:
    write_dir = os.path.join(ROOT_DIR, 'tests', 'logs')
    if not os.path.exists(write_dir):
        os.makedirs
    file_path: str = os.path.join(write_dir, 'test_prompt_{}.txt'.format(str(uuid.uuid4())))  
    with open(file_path, 'w') as file:
        file.write(prompt)


def socket_ip(ip, port, timeout=10) -> bool:
    try:        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)        
        sock.settimeout(timeout)        
        sock.connect((ip, port))
        return True
    except ConnectionRefusedError:
        print(f"Connection refused to {ip}:{port}")
        return False
    except socket.timeout:
        print(f"Connection timeout to {ip}:{port}")
        return False
    except Exception as e:
        print(f"Error connecting to {ip}:{port} - {e}")
        return False

    finally:        
        if 'sock' in locals():
            sock.close()

