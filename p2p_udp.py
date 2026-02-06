from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class PeerPacket:
    sender: str
    phase: str
    ready: int
    sim_time: float
    pos: Tuple[float, float]
    vel: Tuple[float, float]
    recv_wall_time: float


class UdpPeerBus:
    """
    Lightweight localhost UDP broadcaster used to emulate per-robot P2P state exchange.
    One socket is bound per robot name at (host, base_port + idx).
    """

    def __init__(self, robot_names: Iterable[str], host: str = "127.0.0.1", base_port: int = 39000) -> None:
        self.robot_names = list(robot_names)
        self.host = host
        self.base_port = int(base_port)
        self._sockets: Dict[str, socket.socket] = {}
        self._destinations: Dict[str, List[Tuple[str, int]]] = {}
        self._latest: Dict[str, PeerPacket] = {}
        self._name_to_port: Dict[str, int] = {}
        self._bind()

    def _bind(self) -> None:
        for idx, name in enumerate(self.robot_names):
            port = self.base_port + idx
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, port))
            sock.setblocking(False)
            self._sockets[name] = sock
            self._name_to_port[name] = port

        for name in self.robot_names:
            self._destinations[name] = [
                (self.host, self._name_to_port[other])
                for other in self.robot_names
                if other != name
            ]

    def close(self) -> None:
        for sock in self._sockets.values():
            try:
                sock.close()
            except Exception:
                pass
        self._sockets.clear()
        self._destinations.clear()
        self._latest.clear()

    def send(
        self,
        sender: str,
        phase: str,
        ready: int,
        sim_time: float,
        pos_xy: Tuple[float, float],
        vel_xy: Tuple[float, float],
    ) -> None:
        sock = self._sockets.get(sender)
        if sock is None:
            return
        payload = {
            "sender": sender,
            "phase": phase,
            "ready": int(ready),
            "sim_time": float(sim_time),
            "pos": [float(pos_xy[0]), float(pos_xy[1])],
            "vel": [float(vel_xy[0]), float(vel_xy[1])],
        }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        for dest in self._destinations.get(sender, []):
            try:
                sock.sendto(raw, dest)
            except OSError:
                continue

    def poll(self) -> Dict[str, PeerPacket]:
        now = time.time()
        for sock in self._sockets.values():
            while True:
                try:
                    data, _addr = sock.recvfrom(8192)
                except BlockingIOError:
                    break
                except OSError:
                    break
                try:
                    payload = json.loads(data.decode("utf-8"))
                    sender = str(payload.get("sender", "")).strip()
                    if not sender:
                        continue
                    pkt = PeerPacket(
                        sender=sender,
                        phase=str(payload.get("phase", "")),
                        ready=int(payload.get("ready", 0)),
                        sim_time=float(payload.get("sim_time", 0.0)),
                        pos=(float(payload.get("pos", [0.0, 0.0])[0]), float(payload.get("pos", [0.0, 0.0])[1])),
                        vel=(float(payload.get("vel", [0.0, 0.0])[0]), float(payload.get("vel", [0.0, 0.0])[1])),
                        recv_wall_time=now,
                    )
                except Exception:
                    continue

                old = self._latest.get(sender)
                if old is None or pkt.sim_time >= old.sim_time:
                    self._latest[sender] = pkt
        return dict(self._latest)

