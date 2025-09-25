import random
from simulation_env import Node, DataPacket
from strategies import ShortestTtlFirstStrategy, FifoStrategy

# --- シミュレーションの基本設定 ---
SIMULATION_STEPS = 1000        # シミュレーションの総ステップ数
NODE_BANDWIDTH = 200           # 1ステップあたりにノードが転送できる合計サイズ
#  PACKET_ARRIVAL_PROB = 0.95  # 各ステップで新しいデータが到着する確率
MAX_PACKETS_PER_STEP = 100     # 1ステップあたりに到着するデータの最大数
# BUFFER_SIZE_LIMIT = 20       # バッファが保持できる最大のパケット数
BUFFER_BYTE_LIMIT = 500        # バッファが保持できる最大合計サイズ
PACKET_SIZE_RANGE = (1, 10)    # データサイズの範囲 (min, max)
PACKET_TTL_RANGE = (1, 20)     # TTLの範囲 (min, max)

# """セットA"""
# SIMULATION_STEPS = 1000        # シミュレーションの総ステップ数
# NODE_BANDWIDTH = 200           # 1ステップあたりにノードが転送できる合計サイズ
# #  PACKET_ARRIVAL_PROB = 0.95  # 各ステップで新しいデータが到着する確率
# MAX_PACKETS_PER_STEP = 100     # 1ステップあたりに到着するデータの最大数
# # BUFFER_SIZE_LIMIT = 20       # バッファが保持できる最大のパケット数
# BUFFER_BYTE_LIMIT = 500        # バッファが保持できる最大合計サイズ
# PACKET_SIZE_RANGE = (1, 10)    # データサイズの範囲 (min, max)
# PACKET_TTL_RANGE = (1, 20)     # TTLの範囲 (min, max)

""" シミュレーション全体を実行するメイン関数 """
def run_simulation(strategy_class, title):
    # 引数で渡された戦略クラスのインスタンスを作成
    strategy = strategy_class()
    # Nodeの初期化時に戦略インスタンスを渡す
    # Nodeの初期化時に合計サイズの上限を渡す
    node = Node(bandwidth = NODE_BANDWIDTH,
                buffer_byte_limit = BUFFER_BYTE_LIMIT,
                strategy = strategy)

    total_generated = 0
    total_dropped = 0
    all_transmitted = []
    all_expired = []
    
    next_packet_id = 0

    print(f"--- {title} ---")
    for step in range(SIMULATION_STEPS):
        num_new_packets = random.randint(0, MAX_PACKETS_PER_STEP)
        for _ in range(num_new_packets):
            size = random.randint(*PACKET_SIZE_RANGE)
            ttl = random.randint(*PACKET_TTL_RANGE)
            new_packet = DataPacket(packet_id=next_packet_id, size=size, ttl=ttl)
            total_generated += 1
            next_packet_id += 1
            if not node.add_packet(new_packet):
                total_dropped += 1

        # ログ表示部分を修正
        if step % 100 == 0:
            buffer_load = node.get_current_buffer_load()
            print(f"ステップ: {step}, バッファ使用量: {buffer_load}/{BUFFER_BYTE_LIMIT}, "
                  f"総転送数: {len(all_transmitted)}, 総TTL切れ数: {len(all_expired)}")

        transmitted, expired = node.process_step()
        all_transmitted.extend(transmitted)
        all_expired.extend(expired)

    print(f"総生成データ数: {total_generated}")
    print(f"総転送成功データ数: {len(all_transmitted)}")
    print(f"総TTL切れデータ数: {len(all_expired)}")
    print(f"総バッファ溢れによる破棄数: {total_dropped}")
    success_rate_Total = (len(all_transmitted) / total_generated) * 100
    print(f"転送成功率（総生成データ基準）: {success_rate_Total:.2f}%")



if __name__ == "__main__":
    # 最小TTL優先戦略でシミュレーションを実行
    run_simulation(strategy_class = ShortestTtlFirstStrategy, title = "最小TTL優先戦略")

    print("\n" + "="*50 + "\n")

    # FIFO戦略でシミュレーションを実行
    run_simulation(strategy_class = FifoStrategy, title = "FIFO（先入れ先出し）戦略")