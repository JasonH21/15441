import sys
import uuid
import socket
import time
from threading import Thread, Lock

class Peer:
    def __init__(self, uuid, hostname, backend_port, distance_metric):
        self.uuid = uuid
        self.hostname = hostname
        self.backend_port = int(backend_port)
        self.distance_metric = int(distance_metric)

    def to_dict(self):
        return {
            "uuid": self.uuid,
            "host": self.hostname,
            "backend_port": self.backend_port,
            "metric": self.distance_metric
        }

def parse_conf(file_path):
    config = {"peers": {}}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if '=' in line:
                key, value = map(str.strip, line.split('=', 1))
                if key == "peer_count":
                    config[key] = int(value)
                    continue
                if key.startswith("peer_"):
                    peer_data = value.split(", ")
                    if len(peer_data) == 4:
                        peer_uuid, hostname, backend_port, distance_metric = peer_data
                        newKey = "node" + key[5:]
                        config["peers"][newKey] = Peer(peer_uuid, hostname, backend_port, distance_metric)
                else:
                    config[key] = value if not value.isdigit() else int(value)
    return config

def send_keepalive_messages(config, active_neighbors, lock):
    while True:
        for peer_name, peer in config["peers"].items():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.sendto(b"keepalive", (peer.hostname, peer.backend_port))
                with lock:
                    active_neighbors[peer_name] = 0
            except Exception:
                pass
        time.sleep(3)

def receive_keepalive_messages(config, active_neighbors, lock):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("0.0.0.0", config["backend_port"]))
        while True:
            data, addr = sock.recvfrom(1024)
            if data == b"keepalive":
                with lock:
                    for peer_name, peer in config["peers"].items():
                        if (peer.hostname, peer.backend_port) == addr:
                            active_neighbors[peer_name] = 0

def check_neighbor_status(config, active_neighbors, lock):
    while True:
        time.sleep(3)
        with lock:
            for peer_name in list(active_neighbors.keys()):
                active_neighbors[peer_name] += 1
                if active_neighbors[peer_name] > 3:
                    del active_neighbors[peer_name]

def start_command_server(config, active_neighbors, lock):
    while True:
        user_input = input().strip()
        if user_input == "uuid":
            print({"uuid": config["uuid"]})
        elif user_input == "neighbors":
            with lock:
                active_list = {"neighbors": {}}
                
                for name, peer in config["peers"].items():
                    if name in active_neighbors:
                        active_list["neighbors"][peer.hostname] = peer.to_dict()
                        
            print(active_list)


def main():
    conf_path = sys.argv[1]
    config = parse_conf(conf_path)


    if "uuid" not in config:
        config["uuid"] = str(uuid.uuid4())

    active_neighbors = {}
    lock = Lock()

    Thread(target=send_keepalive_messages, args=(config, active_neighbors, lock)).start()
    Thread(target=receive_keepalive_messages, args=(config, active_neighbors, lock)).start()
    Thread(target=check_neighbor_status, args=(config, active_neighbors, lock)).start()
    Thread(target=start_command_server, args=(config, active_neighbors, lock)).start()

if __name__ == "__main__":
    main()
