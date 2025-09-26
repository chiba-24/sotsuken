import math
import numpy as np
# from configs.experiment_configs import DqnTrainConfig
import matplotlib.pyplot as plt

def calculate_shannon_capacity(current_step, config):
    """
    シャノン＝ハートレイの定理とFSPLに基づき、リンク容量を計算する。
    
    Args:
        current_step (int): 現在のシミュレーションステップ。
        config: 必要なパラメータをすべて含む設定オブジェクト。

    Returns:
        float: 計算されたリンク容量 (Mbps)。
    """
    # 物理定数
    BOLTZMANN_CONSTANT = 1.38e-23

    # --- 軌道と距離の計算 ---
    # 軌道半径の計算
    r_geo = config.GEO_ALTITUDE_KM + config.EARTH_RADIUS_KM
    r_leo = config.LEO_ALTITUDE_KM + config.EARTH_RADIUS_KM
    # LEO衛星が円軌道上のどの位置にいるかを計算し，直交座標を計算
    angle_rad = (2 * math.pi * current_step) / config.LEO_ORBITAL_PERIOD_STEPS
    leo_x = r_leo * math.cos(angle_rad)
    leo_y = r_leo * math.sin(angle_rad)
    # GEO衛星の位置を定義し，2衛星の直線距離を計算．
    geo_x = r_geo
    distance_km = math.sqrt((geo_x - leo_x)**2 + leo_y**2)

    # --- FSPLの計算 ---
    # 単位変換
    distance_m = distance_km * 1000
    frequency_hz = config.FREQUENCY_GHz * 1e9
    # FSPLをデシベル単位で計算
    fspl_db = 20 * math.log10(distance_m) + 20 * math.log10(frequency_hz) - 147.55

    # --- 受信信号電力 S の計算 ---
    # 送信電力[W]をデシベルワットに変換
    transmit_power_dbw = 10 * math.log10(config.TRANSMIT_POWER_W)
    # 受信電力 = 送信電力 + 送信アンテナ利得 + 受信アンテナ利得 - 伝搬損失
    received_power_dbw = (transmit_power_dbw + config.TRANSMIT_ANTENNA_GAIN_dBi + 
                          config.RECEIVE_ANTENNA_GAIN_dBi - fspl_db)
    # [dBW]→[W]
    s_watts = 10**(received_power_dbw / 10)

    # --- ノイズ電力 N の計算 ---
    # ノイズ電力 = ボルツマン定数 × 雑音温度 ×　チャネル帯域幅
    channel_bandwidth_hz = config.CHANNEL_BANDWIDTH_MHz * 1e6
    n_watts = BOLTZMANN_CONSTANT * config.SYSTEM_NOISE_TEMPERATURE_K * channel_bandwidth_hz
    
    # --- リンク容量 C の計算 ---
    snr = s_watts / n_watts
    capacity_bps = channel_bandwidth_hz * np.log2(1 + snr)
    
    return capacity_bps / 1e6 # Mbpsに変換して返す