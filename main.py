import random
from config import DqnTrainConfig, ConfigA
from simulation_env import Node # 統一されたNodeクラスをインポート
from strategies import ShortestTtlFirstStrategy, FifoStrategy

def run_simulation(config, strategy_class, title_prefix):
    strategy = strategy_class()
    env = Node(config = config) # 環境としてNodeを初期化

    title = f"{title_prefix} - {config.NAME}"
    stats = {"generated": 0, "transmitted": 0, "expired": 0, "dropped": 0}

    print(f"--- {title} ---")
    env.reset()

    # 外側ループ：時間を進行
    for step in range(config.SIMULATION_STEPS):
        # 1. 時間を1ステップ進め，その間の統計を取得．
        _, time_stats = env.update_time(current_step = step)

        # 全体の統計に加算
        for key in ["generated", "expired", "dropped"]:
            stats[key] += time_stats[key]

        # 内側ループ：帯域幅が尽きるまでパケットを転送
        while env.remaining_bandwidth > 0 and env.buffer:
            # 戦略に行動を選択させる
            packet_to_send = strategy.select_packet(env.buffer)
            action = None
            if packet_to_send:
                try:
                    action = list(env.buffer).index(packet_to_send)
                except ValueError:
                    action = None
            
            # 転送を試みる
            _, transmitted_count, success = env.transmit_packet(action)
            stats["transmitted"] += transmitted_count
            
            # 転送に失敗した（送れるものがない）ら内側ループを抜ける
            if not success:
                break

        # # 2. 従来手法の戦略で転送すべきパケットを選択
        # packet_to_send = strategy.select_packet(env.buffer)
        
        # # 3. 選択したパケットのインデックスを取得してactionとする
        # action = None
        # if packet_to_send:
        #     try:
        #         action = list(env.buffer).index(packet_to_send)
        #     except ValueError:
        #         action = None # すでにTTL切れなどで消えた場合
        
        # # 4. 環境を1ステップ進める
        # _, _, _, step_stats = env.step(action)
        
        # # 5. 統計を更新
        # for key in stats:
        #     stats[key] += step_stats[key]

    print(f"総生成データ数: {stats['generated']}")
    print(f"総転送成功データ数: {stats['transmitted']}")
    print(f"総TTL切れデータ数: {stats['expired']}")
    print(f"総バッファ溢れによる破棄数: {stats['dropped']}")
    if stats['generated'] > 0:
        success_rate = (stats['transmitted'] / stats['generated']) * 100
        print(f"転送成功率: {success_rate:.2f}%")

if __name__ == "__main__":
    config_to_run = DqnTrainConfig()
    run_simulation(config = config_to_run,
                   strategy_class = ShortestTtlFirstStrategy,
                   title_prefix = "最小TTL優先戦略")
    print("\n" + "="*50 + "\n")
    run_simulation(config = config_to_run,
                   strategy_class = FifoStrategy,
                   title_prefix = "FIFO戦略")

# def run_simulation(config, strategy_class, title_prefix):
#     """従来手法のシミュレーションを実行する関数"""
#     strategy = strategy_class()
#     node = Node(bandwidth=config.NODE_BANDWIDTH,
#                 buffer_limit=20, # 簡易版のため固定値
#                 strategy=strategy)

#     title = f"{title_prefix} - {config.NAME}"
#     total_generated, total_dropped, all_transmitted, all_expired = 0, 0, [], []
#     next_packet_id = 0

#     print(f"--- {title} ---")
#     for step in range(config.SIMULATION_STEPS):
#         num_new_packets = random.randint(0, config.MAX_PACKETS_PER_STEP)
#         for _ in range(num_new_packets):
#             size = random.randint(*config.PACKET_SIZE_RANGE)
#             ttl = random.randint(*config.PACKET_TTL_RANGE)
#             new_packet = DataPacket(next_packet_id, size, ttl)
#             total_generated += 1; next_packet_id += 1
#             if not node.add_packet(new_packet): total_dropped += 1
        
#         transmitted, expired = node.process_step()
#         all_transmitted.extend(transmitted); all_expired.extend(expired)

#     print(f"総生成データ数: {total_generated}")
#     print(f"総転送成功データ数: {len(all_transmitted)}")
#     print(f"総TTL切れデータ数: {len(all_expired)}")
#     print(f"総バッファ溢れによる破棄数: {total_dropped}")
#     if total_generated > 0:
#         success_rate = (len(all_transmitted) / total_generated) * 100
#         print(f"転送成功率: {success_rate:.2f}%")


# if __name__ == "__main__":
#     # 実行したい設定を選択
#     config_to_run = ConfigA()
    
#     print("実行モード: 従来手法の比較シミュレーション")
    
#     run_simulation(config=config_to_run,
#                    strategy_class=ShortestTtlFirstStrategy,
#                    title_prefix="最小TTL優先戦略")
#     print("\n" + "="*50 + "\n")
#     run_simulation(config=config_to_run,
#                    strategy_class=FifoStrategy,
#                    title_prefix="FIFO戦略")