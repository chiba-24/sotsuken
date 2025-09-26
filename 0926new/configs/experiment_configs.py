class BaseConfig:
    """
    全ての実験シナリオに共通する基本設定。
    個別の設定クラスは、このクラスを継承して作成する。
    """
    # シミュレーション全体のステップ数
    SIMULATION_STEPS = 100
    
    # パケットの特性
    PACKET_SIZE_RANGE = (5, 10)
    PACKET_TTL_RANGE = (5, 5)



    # 物理・軌道パラメータ
    EARTH_RADIUS_KM = 6371
    GEO_ALTITUDE_KM = 35786
    LEO_ALTITUDE_KM = 550
    LEO_ORBITAL_PERIOD_STEPS = 2000 # LEO衛星が1周するステップ数
    NUM_LEOS_PER_ORBIT = 4 # 1軌道あたりのLEO衛星の数

    # リンクバジェット（送受信システム）のパラメータ
    TRANSMIT_POWER_W = 10.0      # 送信電力 (W)
    TRANSMIT_ANTENNA_GAIN_dBi = 40.0 # 送信アンテナ利得 (dBi)
    RECEIVE_ANTENNA_GAIN_dBi = 40.0  # 受信アンテナ利得 (dBi)
    FREQUENCY_GHz = 12.0         # 周波数 (GHz) - Kuバンドを想定
    CHANNEL_BANDWIDTH_MHz = 500.0  # チャネル帯域幅 (MHz)
    SYSTEM_NOISE_TEMPERATURE_K = 150.0 # システム雑音温度 (K)

# class ConfigA(BaseConfig):
#     """
#     設定セットA: 標準的なテストシナリオ
#     """
#     NAME = "Config A (標準シナリオ)"
    
#     # 環境パラメータ
#     MAX_BANDWIDTH = 150         # 最大帯域幅
#     MAX_PACKETS_PER_STEP = 3    # 1ステップあたりの最大到着パケット数
#     BUFFER_PACKET_LIMIT = 20    # バッファの最大パケット収容数

# class ConfigB(BaseConfig):
#     """
#     設定セットB: 高トラフィック・シナリオ
#     """
#     NAME = "Config B (高トラフィック)"
    
#     # 環境パラメータ
#     MAX_BANDWIDTH = 150
#     MAX_PACKETS_PER_STEP = 5    # ← 標準よりパケット到着数を多くする
#     BUFFER_PACKET_LIMIT = 20

class DqnTrainConfig(BaseConfig):
    """
    DQNの学習フェーズ専用の設定。
    学習用のステップ数を長くしたり、DQNのハイパーパラメータを定義する。
    """
    NAME = "DQN Training Config"
    
    # 学習用にシミュレーションステップを長くする
    SIMULATION_STEPS = 50000
    
    # # 環境パラメータ (標準シナリオと同じか、学習に適した値に設定)
    MAX_BANDWIDTH = 150
    MAX_PACKETS_PER_STEP = 20
    BUFFER_PACKET_LIMIT = 20000
    BUFFER_BYTE_LIMIT = 100



    # --- DQN専用ハイパーパラメータ ---
    HIDDEN_LAYER_SIZES = [128, 128]
    REPLAY_BUFFER_CAPACITY = 10000
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 20000
    TARGET_UPDATE_FREQUENCY = 15