from collections import deque


class DataPacket:
    def __init__(self, packet_id, size, ttl):
        self.id = packet_id
        self.size = size
        self.ttl = ttl
        self.initial_ttl = ttl

    def decrement_ttl(self):
        self.ttl -= 1

    def __repr__(self):
        return f"Packet(ID:{self.id}, Size:{self.size}, TTL:{self.ttl}/{self.initial_ttl})"

""" データパケットを保持し、転送を管理するノードのクラス """
class Node:
    def __init__(self, bandwidth, buffer_byte_limit, strategy):
        self.bandwidth = bandwidth
        self.buffer_byte_limit = buffer_byte_limit # 合計サイズの上限
        self.strategy = strategy
        self.buffer = deque()

    """現在のバッファ内の合計サイズを計算して返す"""
    def get_current_buffer_load(self):
        return sum(packet.size for packet in self.buffer)

    """バッファに新しいパケットを追加する。合計サイズ上限を超えていたら追加しない。"""
    def add_packet(self, packet):
        # 新しいパケットを追加しても上限を超えないかチェック
        if self.get_current_buffer_load() + packet.size <= self.buffer_byte_limit:
            self.buffer.append(packet)
            return True  # 追加成功
        else:
            return False # 追加失敗（バッファが満杯）

    """1タイムステップ分の処理を進める"""
    def process_step(self):
        transmitted_packets = []
        expired_packets = []

        for packet in list(self.buffer):
            packet.decrement_ttl()
            if packet.ttl <= 0:
                self.buffer.remove(packet)
                expired_packets.append(packet)

        remaining_bandwidth = self.bandwidth
        while remaining_bandwidth > 0 and self.buffer:
            # 外部から注入された戦略オブジェクトにパケット選択を「委任」する
            packet_to_send = self.strategy.select_packet(self.buffer)

            if packet_to_send is None:
                break
            if packet_to_send.size <= remaining_bandwidth:
                remaining_bandwidth -= packet_to_send.size
                self.buffer.remove(packet_to_send)
                transmitted_packets.append(packet_to_send)
            else:
                break
        return transmitted_packets, expired_packets