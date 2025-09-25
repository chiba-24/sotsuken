class BaseConfig:
    SIMULATION_STEPS = 10000  # DQNの学習には多くのステップが必要
    PACKET_SIZE_RANGE = (10, 50)
    PACKET_TTL_RANGE = (5, 20)

class ConfigA(BaseConfig):
    """
    設定セットA: 標準的なトラフィック
    """
    NAME = "Config A (標準)"
    NODE_BANDWIDTH = 100
    MAX_PACKETS_PER_STEP = 3
    BUFFER_BYTE_LIMIT = 500

class ConfigB(BaseConfig):
    """
    設定セットB: 高トラフィック・小バッファ
    """
    NAME = "Config B (高トラフィック)"
    NODE_BANDWIDTH = 100
    MAX_PACKETS_PER_STEP = 5  # 到着するパケット数を増やす
    BUFFER_BYTE_LIMIT = 300  # バッファを小さくする

class ConfigC(BaseConfig):
    """
    設定セットC: 低帯域幅
    """
    NAME = "Config C (低帯域幅)"
    NODE_BANDWIDTH = 50      # ノードの転送能力を半分にする
    MAX_PACKETS_PER_STEP = 3
    BUFFER_BYTE_LIMIT = 500


class DqnTrainConfig(BaseConfig):
    NAME = "DQN Training"
    NODE_BANDWIDTH = 100
    MAX_PACKETS_PER_STEP = 2
    BUFFER_PACKET_LIMIT = 20    # バッファの最大パケット数を固定
    BUFFER_BYTE_LIMIT = 1000    # こちらは参考値とする

    # --- DQN Hyperparameters ---
    GAMMA = 0.99                # 割引率
    EPSILON_START = 0.9         # εの初期値
    EPSILON_END = 0.05          # εの最終値
    EPSILON_DECAY = 10000       # εの減衰速度
    LEARNING_RATE = 1e-4        # 学習率
    BATCH_SIZE = 128            # バッチサイズ
    TARGET_UPDATE_FREQUENCY = 15 # ターゲットネットワークの更新頻度 (ステップ数)
