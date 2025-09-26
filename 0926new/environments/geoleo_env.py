import math
import random
from collections import deque
import numpy as np

# 同じフォルダにあるbase_envからBaseEnvをインポート
from .base_env import BaseEnv
# utilsフォルダから、分離したリンク容量の計算関数をインポート
from utils.link_models import calculate_shannon_capacity

class DataPacket:
    """
    シミュレーション内で扱われる個々のデータパケットの情報。
    このファイル内で閉じて使うため、ここに定義する。
    """
    def __init__(self, packet_id, size, ttl):
        self.id = packet_id
        self.size = size
        self.ttl = ttl
    def __repr__(self):
        return f"P(id:{self.id},size:{self.size},ttl:{self.ttl})"

class GeoLeoEnv(BaseEnv):
    """
    GEO-LEO衛星間のリンク容量変動をモデル化した具体的なシミュレーション環境。
    """
    def __init__(self, config):
        self.config = config
        self.buffer = deque()
        self.packet_id_counter = 0
        
        # 初期帯域幅を設定（configに最大値があればそれ、なければ中心値）
        self.remaining_bandwidth = getattr(config, 'MAX_BANDWIDTH', getattr(config, 'BANDWIDTH_CENTER', 100))

    def reset(self):
        """環境を初期状態にリセットする"""
        self.buffer.clear()
        self.packet_id_counter = 0

        # リセット時も初期帯域幅を設定
        self.remaining_bandwidth = getattr(self.config, 'MAX_BANDWIDTH', getattr(self.config, 'BANDWIDTH_CENTER', 100))
        return self.get_state()

    def update_time(self, current_step):
        """時間が1ステップ進んだ際の、環境の自動的な変化を処理する"""
        generated_count, expired_count, dropped_count = 0, 0, 0
        expired_reward = 0

        # --- 帯域幅の計算 ---
        # 複雑な計算は外部のlink_models.pyに委任
        current_bandwidth = calculate_shannon_capacity(current_step, self.config)
        self.remaining_bandwidth = int(current_bandwidth)

        # --- パケット到着とTTL減少 ---
        # 1. 新しいパケットの到着
        num_new_packets = random.randint(0, self.config.MAX_PACKETS_PER_STEP)
        for _ in range(num_new_packets):
            size = random.randint(*self.config.PACKET_SIZE_RANGE)
            ttl = random.randint(*self.config.PACKET_TTL_RANGE)
            new_packet = DataPacket(self.packet_id_counter, size, ttl)
            self.packet_id_counter += 1
            generated_count += 1

            # 現在のバッファ内の合計サイズを計算
            current_buffer_load = sum(p.size for p in self.buffer)

            # パケット数と合計サイズの両方の上限をチェック
            if (len(self.buffer) < self.config.BUFFER_PACKET_LIMIT and
                current_buffer_load + new_packet.size <= self.config.BUFFER_BYTE_LIMIT):
                
                self.buffer.append(new_packet) # 条件を満たせば追加
            else:
                dropped_count += 1 # どちらかの上限に達していれば破棄
        
        # 2. TTLの減少と期限切れの確認
        for packet in list(self.buffer):
            packet.ttl -= 1
            if packet.ttl <= 0:
                self.buffer.remove(packet)
                expired_count += 1
        
        expired_reward -= expired_count * 100
        
        stats = {"generated": generated_count, "expired": expired_count, "dropped": dropped_count}
        return expired_reward, stats

    def transmit_packet(self, action):
        """エージェントから受け取ったactionを処理する"""
        if action is None or not (0 <= action < len(self.buffer)):
            return -20, 0, False # 罰則, 転送数, 成功フラグ
        
        packet_to_send = self.buffer[action]
        if packet_to_send.size <= self.remaining_bandwidth:
            self.remaining_bandwidth -= packet_to_send.size
            self.buffer.remove(packet_to_send)
            return 10, 1, True # 報酬, 転送数, 成功フラグ
        else:
            return -5, 0, False # 罰則, 転送数, 成功フラグ

    def get_state(self):
        """現在の環境の状態を、エージェントが理解できる形式で返す"""
        state = np.zeros((self.config.BUFFER_PACKET_LIMIT, 2), dtype=np.float32)
        for i, packet in enumerate(self.buffer):
            if i >= self.config.BUFFER_PACKET_LIMIT: break
            # TTLとサイズを正規化して状態表現とする
            state[i, 0] = packet.ttl / self.config.PACKET_TTL_RANGE[1]
            state[i, 1] = packet.size / self.config.PACKET_SIZE_RANGE[1]
        return state.flatten()