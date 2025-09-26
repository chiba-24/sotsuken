from collections import deque
import random
from collections import deque
import numpy as np
import math

class DataPacket:
    def __init__(self, packet_id, size, ttl):
        self.id = packet_id
        self.size = size
        self.ttl = ttl
        self.initial_ttl = ttl # 初期TTL

    # TTLを-1する関数
    def decrement_ttl(self):
        self.ttl -= 1

    # オブジェクトの「公式な」文字列表現を定義
    # 例：「Packet(ID:10，Size:35, TTL:5/10)」
    def __repr__(self):
        return f"Packet(ID:{self.id}, Size:{self.size}, TTL:{self.ttl}/{self.initial_ttl})"

""" ノードの環境クラス """
class Node:
    """ オブジェクトの初期設定 """
    def __init__(self, config):
        self.config = config
        # データパケットを格納するためのバッファ（待ち行列）をdequeで作成
        self.buffer = deque()
        # 新規パケットにIDを割り振るためのカウンタを初期化
        self.packet_id_counter = 0
        # 帯域幅の中心
        self.remaining_bandwidth = self.config.BANDWIDTH_CENTER

    """ 現在のバッファの状態をNNが理解できる固定長の数値リストに変換 """
    def _get_state(self):
        # 中身がすべて0のNumpy配列を作成．大きさは「最大パケット数 × 2（TTLとサイズの2つの特徴量）」
        state = np.zeros((self.config.BUFFER_PACKET_LIMIT, 2), dtype = np.float32)

        # バッファ内の各パケットに対して，そのインデックスiとパケット本体をセットで取り出すループ処理
        for i, packet in enumerate(self.buffer):
            # バッファ内のパケット数が定義した最大数を超えていた場合に，配列の範囲外アクセスを防ぐための安全装置
            if i >= self.config.BUFFER_PACKET_LIMIT: break
            # i番目のパケットのTTLを，設定された最大TTLで割って正規化（0〜1の範囲にスケーリング）した値．
            state[i, 0] = packet.ttl / self.config.PACKET_TTL_RANGE[1]
            # i番目のパケットのサイズを，設定された最大サイズで割って正規化した値
            state[i, 1] = packet.size / self.config.PACKET_SIZE_RANGE[1]
        # 2次元配列を1次元配列に変換
        return state.flatten()

    """ シミュレーション環境を初期化 """
    def reset(self):
        # ノードのバッファリストを初期化
        self.buffer.clear()
        # パケットのIDカウンタを初期化
        self.packet_id_counter = 0
        # 帯域幅の中心
        self.remaining_bandwidth = self.config.BANDWIDTH_CENTER
        # バッファリストをNNが読み込める数値リストに変換
        return self._get_state()

    """ エージェントから行動を受け取り，時間が1ステップ進んだ時の環境の変化を計算 """
    def update_time(self, current_step):
        # # 生存ペナルティとして，行動するたびに報酬を減少．
        # reward = -1
        # 各カウンタを初期化
        generated_count = 0
        # transmitted_count = 0
        expired_count = 0
        dropped_count = 0
        expired_reward = 0
        
        # 1．帯域幅をサイン波に基づいて周期的に設定
        center = self.config.BANDWIDTH_CENTER
        amplitude = self.config.BANDWIDTH_AMPLITUDE
        period = self.config.BANDWIDTH_PERIOD
        # サイン波の計算 (-1.0 ~ 1.0 の値を生成)
        oscillation = math.sin(2 * math.pi * current_step / period)
        # 最終的な帯域幅を計算
        self.remaining_bandwidth = int(center + amplitude * oscillation)

        # 2. 新規パケットの到来．
        # 0~最大数の間でランダムに決定．
        num_new_packets = random.randint(0, self.config.MAX_PACKETS_PER_STEP)
        # ランダムなサイズとTTLを持つ新規パケットを生成．IDと生成カウンタを1加算．
        for _ in range(num_new_packets):
            size = random.randint(*self.config.PACKET_SIZE_RANGE)
            ttl = random.randint(*self.config.PACKET_TTL_RANGE)
            new_packet = DataPacket(self.packet_id_counter, size, ttl)
            self.packet_id_counter += 1
            generated_count += 1

            # バッファが満杯ならパケットを破棄
            if len(self.buffer) < self.config.BUFFER_PACKET_LIMIT:
                self.buffer.append(new_packet)
            else:
                dropped_count += 1
        
        # 3. TTLの減少と期限切れの確認
        for packet in list(self.buffer):
            # TTLを減少
            packet.ttl -= 1
            if packet.ttl <= 0:
                # TTLが0以下のパケットは破棄して，破棄カウンタを1加算．
                self.buffer.remove(packet)
                expired_count += 1
        # パケット損失数に応じて，報酬を大きく減少．
        expired_reward -= expired_count * 100

        # # 3. 指定されたactionを処理
        # # エージェントが選択した行動（転送したいパケットのインデックス）が，
        # # 現在のバッファリストに存在するか確認．
        # if action is not None and 0 <= action < len(self.buffer):
        #     # 指定されたパケットを取り出す．
        #     packet_to_send = self.buffer[action]
        #     # 転送したいパケットのサイズが，ノードの転送能力以下であるかを確認．
        #     if packet_to_send.size <= self.config.NODE_BANDWIDTH:
        #         # 転送可能ならパケットをバッファから削除．
        #         self.buffer.remove(packet_to_send)
        #         # 正の報酬を獲得．
        #         reward += 10
        #         # 転送成功カウンタを1加算．
        #         transmitted_count = 1
        #     else:
        #         # 帯域幅よりも大きいパケットを送ろうとした場合には負の報酬を獲得．
        #         reward -= 5
        # elif action is not None:
        #      reward -= 20 # 無効なアクションに対する報酬．

        # すべての処理が終わった後の，新バッファの状態を数値リストに変換．
        # next_state = self._get_state()
        # 終了したらTrueを返すが，この環境では通常終了しない．
        # done = False
        # このステップで発生したイベントの統計を辞書に保存．
        stats = {
            "generated": generated_count, "expired": expired_count, "dropped": dropped_count
        }
        return expired_reward, stats
    
    


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