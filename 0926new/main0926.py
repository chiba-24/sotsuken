# 実験シナリオ設定
from configs.experiment_configs import DqnTrainConfig 

# シミュレーション環境
from environments.geoleo_env import GeoLeoEnv

# 比較手法と提案手法（戦略）
from strategies.simple_strategies import FifoStrategy, ShortestTtlFirstStrategy
# from strategies.dqn_strategy import DqnStrategy


def run_experiment(config):
    """
    一つの設定（config）に基づき、複数の戦略を評価する実験を実行する。
    """
    print(f"=============== 実験開始: {config.NAME} ===============")

    # 実験で使う環境を初期化
    # ----------------------------------------------------
    env = GeoLeoEnv(config)
    # ----------------------------------------------------

    # 3. 比較したい戦略をリストアップ
    # ----------------------------------------------------
    strategies_to_test = [
        ("FIFO Strategy", FifoStrategy),
        ("Shortest TTL First Strategy", ShortestTtlFirstStrategy)
        # ("DQN Strategy", DqnStrategy)
    ]
    # ----------------------------------------------------

    # 4. 全ての戦略の結果を保存するための辞書
    # ----------------------------------------------------
    results = {}
    # ----------------------------------------------------

    # 5. 各戦略を順番にテストするループ
    # ----------------------------------------------------
    for strategy_name, strategy_class in strategies_to_test:
        print(f"\n--- 戦略 '{strategy_name}' の評価を開始 ---")
        
        # 戦略を初期化
        strategy = strategy_class(config) # configを渡す (DQNなどで利用)

        # # 5a. もし戦略がDQNなら、学習フェーズを実行
        # if isinstance(strategy, DqnStrategy):
        #     print("DQNの学習を開始します...")
        #     strategy.train(env) # 学習ループはDQNクラス内にカプセル化
        #     print("DQNの学習が完了しました。")

        # 5b. 評価フェーズ（全戦略で共通）
        print("評価シミュレーションを開始します...")
        env.reset()
        stats = {"transmitted": 0, "expired": 0, "dropped": 0, "generated": 0}
        
        for step in range(config.SIMULATION_STEPS):
            # 時間を進め、環境の変化を処理
            _, time_stats = env.update_time(current_step=step)
            for key in time_stats:
                stats[key] += time_stats[key]
            
            # 帯域幅が尽きるまでパケット転送
            while env.remaining_bandwidth > 0 and env.buffer:
                # 戦略に行動を選択させる（DQNは学習済みモデルで推論）
                action = strategy.select_action(env)
                
                # 転送を試みる
                _, transmitted_count, success = env.transmit_packet(action)
                stats["transmitted"] += transmitted_count
                
                if not success:
                    break
        
        # 5c. 結果を保存
        if stats["generated"] > 0:
            success_rate = (stats["transmitted"] / stats["generated"]) * 100
            results[strategy_name] = success_rate
            print(f"結果: 総生成パケット数 = {stats["generated"]}")
            print(f"　　  転送パケット数　 = {stats["transmitted"]}")
            print(f"　　  破棄パケット数　 = {stats["dropped"]}")
            print(f"　　  転送成功率　　　 = {success_rate:.2f}%")
        else:
            results[strategy_name] = 0
            print("結果: パケットは生成されませんでした。")

    # 6. 最終結果をまとめて表示
    # ----------------------------------------------------
    print("\n=============== 全戦略の最終結果比較 ===============")
    for strategy_name, success_rate in results.items():
        print(f"{strategy_name:<30}: {success_rate:>6.2f}%")
    print("=====================================================")


if __name__ == "__main__":
    # 7. 実行したい実験シナリオを選択
    # ----------------------------------------------------
    # ConfigAで実験を実行
    run_experiment(config = DqnTrainConfig())
    
    # ConfigBで実験したい場合は、以下のコメントを外す
    # print("\n\n")
    # run_experiment(config=ConfigB())