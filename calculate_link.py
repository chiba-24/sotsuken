import math
import matplotlib.pyplot as plt

# --- 1. 基本設定 ---
# このセクションの値を変更することで、様々なシミュレーションが可能です。

# 物理・軌道パラメータ
EARTH_RADIUS_KM = 6371      # 地球の半径 (km)
GEO_ALTITUDE_KM = 35786     # 静止軌道高度 (km)
LEO_ALTITUDE_KM = 550       # 低軌道高度 (km)

# リンクパラメータ
MAX_BANDWIDTH = 150         # 衛星間が最接近した時の最大帯域幅 (例: Mbps)

# シミュレーション設定
SIMULATION_STEPS = 2000     # LEO衛星が1周するのにかかる総ステップ数

def calculate_bandwidth_at_step(current_step):
    """
    指定された時間ステップにおけるGEO-LEO間の帯域幅を計算する関数。
    
    Args:
        current_step (int): 現在のシミュレーションステップ。

    Returns:
        float: 計算された帯域幅。
    """
    # --- 2. 軌道半径と最短距離の計算 ---
    r_geo = GEO_ALTITUDE_KM + EARTH_RADIUS_KM
    r_leo = LEO_ALTITUDE_KM + EARTH_RADIUS_KM
    min_distance = r_geo - r_leo

    # --- 3. LEO衛星の現在位置を計算 (2次元でモデル化) ---
    # 1周期(SIMULATION_STEPS)で2πラジアン(360度)進む
    angle_rad = (2 * math.pi * current_step) / SIMULATION_STEPS
    
    # 地球の中心を(0,0)としたときのLEO衛星の座標
    leo_x = r_leo * math.cos(angle_rad)
    leo_y = r_leo * math.sin(angle_rad)
    
    # --- 4. GEO衛星の位置を定義 ---
    # GEO衛星はX軸上に静止していると仮定
    geo_x = r_geo
    geo_y = 0
    
    # --- 5. 衛星間の距離を計算 ---
    distance_km = math.sqrt((geo_x - leo_x)**2 + (geo_y - leo_y)**2)
    
    # --- 6. 距離に基づいて帯域幅を計算 (簡易FSPLモデル) ---
    # 信号強度は距離の2乗に反比例するという物理法則をモデル化
    # 距離が最短(min_distance)の時に帯域幅が最大(MAX_BANDWIDTH)になる
    if distance_km > 0:
        ratio = (min_distance / distance_km) ** 2
        current_bandwidth = MAX_BANDWIDTH * ratio
    else:
        current_bandwidth = MAX_BANDWIDTH
        
    return current_bandwidth

def main():
    """
    シミュレーションを実行し、結果をプロットするメイン関数
    """
    print("GEO-LEO間リンク容量シミュレーションを開始します。")
    
    timesteps = []
    bandwidths = []
    
    # --- 7. シミュレーションループ ---
    for step in range(SIMULATION_STEPS):
        bandwidth = calculate_bandwidth_at_step(step)
        
        timesteps.append(step)
        bandwidths.append(bandwidth)

        if step % (SIMULATION_STEPS // 4) == 0:
            print(f"ステップ {step:>4}: 帯域幅 = {bandwidth:.2f} Mbps")
            
    print("シミュレーションが完了しました。")

    # --- 8. 結果をグラフで可視化 ---
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, bandwidths)
    plt.title("Bandwidth Variation between GEO and LEO Satellites")
    plt.xlabel("Simulation Step")
    plt.ylabel(f"Bandwidth ({'Mbps' if MAX_BANDWIDTH else ''})")
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()

# このスクリプトが直接実行された場合にmain関数を呼び出す
if __name__ == "__main__":
    main()