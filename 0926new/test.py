import math
import numpy as np
from configs.experiment_configs import ConfigA, DqnTrainConfig
import matplotlib.pyplot as plt
from utils.link_models import calculate_shannon_capacity

# --- ▼▼▼ 変更点 ▼▼▼ ---
# main関数もconfigオブジェクトを引数として受け取る
def main(config):
    """シミュレーションを実行し、結果をプロットする"""
    print(f"--- {config.NAME} の設定でリンク容量シミュレーションを実行 ---")
    
    timesteps = list(range(config.SIMULATION_STEPS))
    # calculate_link_capacityにconfigを渡す
    capacities = [calculate_shannon_capacity(step, config) for step in timesteps]

    for step in timesteps:
        if step % (config.SIMULATION_STEPS // 4) == 0:
            print(f"ステップ {step:>4}: リンク容量 = {capacities[step]:.2f} Mbps")

    print("シミュレーション完了")
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, capacities)
    plt.title(f"Link Capacity ({config.NAME})")
    plt.xlabel("Simulation Step")
    plt.ylabel("Capacity (Mbps)")
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()
# --- ▲▲▲ 変更点 ▲▲▲ ---


if __name__ == "__main__":
    # --- ▼▼▼ 変更点 ▼▼▼ ---
    # 1. ここで実行したい設定クラスを選択する
    config_to_run = ConfigA()
    # config_to_run = DqnTrainConfig() # こちらの行のコメントを外せばDQN用の設定で試せる

    # 2. 選択した設定オブジェクトをmain関数に渡して実行
    main(config=config_to_run)
    # --- ▲▲▲ 変更点 ▲▲▲ ---