from collections import deque
import random
from collections import deque
import numpy as np

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

class NodeEnv:
    def __init__(self, config):
        self.config = config
        self.buffer = deque()
        self.packet_id_counter = 0

    def _get_state(self):
        """現在のバッファの状態を固定長のNumpy配列で返す"""
        state = np.zeros((self.config.BUFFER_PACKET_LIMIT, 2), dtype=np.float32)
        for i, packet in enumerate(self.buffer):
            if i >= self.config.BUFFER_PACKET_LIMIT: break
            # TTLとサイズを正規化して状態表現とする
            state[i, 0] = packet.ttl / self.config.PACKET_TTL_RANGE[1]
            state[i, 1] = packet.size / self.config.PACKET_SIZE_RANGE[1]
        return state.flatten() # 1次元配列に変換

    def reset(self):
        """環境をリセットする"""
        self.buffer.clear()
        self.packet_id_counter = 0
        return self._get_state()

    def step(self, action):
        """エージェントが選択した行動を実行し、結果を返す"""
        # メソッドの最初にカウンター変数を0で初期化する
        reward = -1
        generated_count = 0
        transmitted_count = 0
        expired_count = 0
        
        # 1. 新しいパケットの到着
        num_new_packets = random.randint(0, self.config.MAX_PACKETS_PER_STEP)
        for _ in range(num_new_packets):
            if len(self.buffer) < self.config.BUFFER_PACKET_LIMIT:
                size = random.randint(*self.config.PACKET_SIZE_RANGE)
                ttl = random.randint(*self.config.PACKET_TTL_RANGE)
                new_packet = DataPacket(self.packet_id_counter, size, ttl)
                self.buffer.append(new_packet)
                self.packet_id_counter += 1
                generated_count += 1

        # 2. TTLの減少と期限切れの確認
        for packet in list(self.buffer):
            packet.ttl -= 1
            if packet.ttl <= 0:
                self.buffer.remove(packet)
                expired_count += 1
        reward -= expired_count * 100

        # 3. エージェントの行動を処理
        if action >= len(self.buffer):
            reward -= 20
        else:
            packet_to_send = self.buffer[action]
            if packet_to_send.size <= self.config.NODE_BANDWIDTH:
                self.buffer.remove(packet_to_send)
                reward += 10
                transmitted_count = 1 # 転送成功
            else:
                reward -= 5
        
        next_state = self._get_state()
        done = False
        
        stats = {
            "generated": generated_count,
            "transmitted": transmitted_count,
            "expired": expired_count
        }
        return next_state, reward, done, stats

class SimpleSimulationNode:
    """従来手法のシミュレーションで使うためのシンプルなNodeクラス"""
    def __init__(self, bandwidth, buffer_limit, strategy):
        self.bandwidth = bandwidth
        self.buffer_limit = buffer_limit
        self.strategy = strategy
        self.buffer = [] # dequeではなくシンプルなリストでOK

    def add_packet(self, packet):
        if len(self.buffer) < self.buffer_limit:
            self.buffer.append(packet)
            return True
        return False

    def process_step(self):
        transmitted, expired = [], []
        # TTL減少と期限切れチェック
        for p in self.buffer[:]: # コピーをループ
            p.ttl -= 1
            if p.ttl <= 0:
                self.buffer.remove(p)
                expired.append(p)
        
        # 転送処理
        remaining_bandwidth = self.bandwidth
        while remaining_bandwidth > 0 and self.buffer:
            packet_to_send = self.strategy.select_packet(self.buffer)
            if packet_to_send and packet_to_send.size <= remaining_bandwidth:
                remaining_bandwidth -= packet_to_send.size
                self.buffer.remove(packet_to_send)
                transmitted.append(packet_to_send)
            else:
                # 送れない or 送るものがない
                break
        return transmitted, expired
    
    


# """ データパケットを保持し、転送を管理するノードのクラス """
# class Node:
#     def __init__(self, bandwidth, buffer_byte_limit, strategy):
#         self.bandwidth = bandwidth
#         self.buffer_byte_limit = buffer_byte_limit # 合計サイズの上限
#         self.strategy = strategy
#         self.buffer = deque()

#     """現在のバッファ内の合計サイズを計算して返す"""
#     def get_current_buffer_load(self):
#         return sum(packet.size for packet in self.buffer)

#     """バッファに新しいパケットを追加する。合計サイズ上限を超えていたら追加しない。"""
#     def add_packet(self, packet):
#         # 新しいパケットを追加しても上限を超えないかチェック
#         if self.get_current_buffer_load() + packet.size <= self.buffer_byte_limit:
#             self.buffer.append(packet)
#             return True  # 追加成功
#         else:
#             return False # 追加失敗（バッファが満杯）

#     """1タイムステップ分の処理を進める"""
#     def process_step(self):
#         transmitted_packets = []
#         expired_packets = []

#         for packet in list(self.buffer):
#             packet.decrement_ttl()
#             if packet.ttl <= 0:
#                 self.buffer.remove(packet)
#                 expired_packets.append(packet)

#         remaining_bandwidth = self.bandwidth
#         while remaining_bandwidth > 0 and self.buffer:
#             # 外部から注入された戦略オブジェクトにパケット選択を「委任」する
#             packet_to_send = self.strategy.select_packet(self.buffer)

#             if packet_to_send is None:
#                 break
#             if packet_to_send.size <= remaining_bandwidth:
#                 remaining_bandwidth -= packet_to_send.size
#                 self.buffer.remove(packet_to_send)
#                 transmitted_packets.append(packet_to_send)
#             else:
#                 break
#         return transmitted_packets, expired_packets