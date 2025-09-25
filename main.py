import random
from config import ConfigA, ConfigB
from simulation_env import DataPacket
from strategies import ShortestTtlFirstStrategy, FifoStrategy
# 簡易版シミュレーション用Nodeをインポート
from simulation_env import SimpleSimulationNode as Node 

def run_simulation(config, strategy_class, title_prefix):
    """従来手法のシミュレーションを実行する関数"""
    strategy = strategy_class()
    node = Node(bandwidth=config.NODE_BANDWIDTH,
                buffer_limit=20, # 簡易版のため固定値
                strategy=strategy)

    title = f"{title_prefix} - {config.NAME}"
    total_generated, total_dropped, all_transmitted, all_expired = 0, 0, [], []
    next_packet_id = 0

    print(f"--- {title} ---")
    for step in range(config.SIMULATION_STEPS):
        num_new_packets = random.randint(0, config.MAX_PACKETS_PER_STEP)
        for _ in range(num_new_packets):
            size = random.randint(*config.PACKET_SIZE_RANGE)
            ttl = random.randint(*config.PACKET_TTL_RANGE)
            new_packet = DataPacket(next_packet_id, size, ttl)
            total_generated += 1; next_packet_id += 1
            if not node.add_packet(new_packet): total_dropped += 1
        
        transmitted, expired = node.process_step()
        all_transmitted.extend(transmitted); all_expired.extend(expired)

    print(f"総生成データ数: {total_generated}")
    print(f"総転送成功データ数: {len(all_transmitted)}")
    print(f"総TTL切れデータ数: {len(all_expired)}")
    print(f"総バッファ溢れによる破棄数: {total_dropped}")
    if total_generated > 0:
        success_rate = (len(all_transmitted) / total_generated) * 100
        print(f"転送成功率: {success_rate:.2f}%")


if __name__ == "__main__":
    # 実行したい設定を選択
    config_to_run = ConfigA()
    
    print("実行モード: 従来手法の比較シミュレーション")
    
    run_simulation(config=config_to_run,
                   strategy_class=ShortestTtlFirstStrategy,
                   title_prefix="最小TTL優先戦略")
    print("\n" + "="*50 + "\n")
    run_simulation(config=config_to_run,
                   strategy_class=FifoStrategy,
                   title_prefix="FIFO戦略")