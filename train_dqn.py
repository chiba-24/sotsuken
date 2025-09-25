import torch
import numpy as np

from config import DqnTrainConfig, ConfigA
from simulation_env import NodeEnv
from dqn_agent import DqnAgent

def evaluate_agent(agent, env, eval_steps=10000):
    """学習済みエージェントの性能を評価する"""
    print("\n--- エージェントの性能評価開始 ---")
    
    total_generated = 0
    total_transmitted = 0
    total_expired = 0
    
    state = env.reset()
    
    for _ in range(eval_steps):
        # 評価時はε-greedyを使わず、最適な行動のみを選択
        with torch.no_grad():
            state_tensor = torch.tensor(state, device="cpu", dtype=torch.float32).unsqueeze(0)
            action = agent.policy_net(state_tensor).max(1)[1].item()
            
        next_state, _, _, stats = env.step(action)
        state = next_state
        
        total_generated += stats["generated"]
        total_transmitted += stats["transmitted"]
        total_expired += stats["expired"]

    print("\n--- 評価結果 ---")
    print(f"総生成データ数: {total_generated}")
    print(f"総転送成功データ数: {total_transmitted}")
    print(f"総TTL切れデータ数: {total_expired}")
    
    if total_generated > 0:
        success_rate = (total_transmitted / total_generated) * 100
        print(f"転送成功率: {success_rate:.2f}%")


def train_dqn():
    """DQNの学習を実行するメインループ"""
    config = DqnTrainConfig()
    env = NodeEnv(config)
    
    # 状態と行動の次元数を設定から取得
    state_size = config.BUFFER_PACKET_LIMIT * 2 # TTLとサイズの2つの特徴量
    action_size = config.BUFFER_PACKET_LIMIT
    
    agent = DqnAgent(state_size, action_size, config)
    
    rewards_log = []
    
    print(f"--- {config.NAME} 開始 ---")
    state = env.reset()
    
    for step in range(1, config.SIMULATION_STEPS + 1):
        # 1. エージェントが行動を選択
        action_tensor = agent.select_action(state)
        action = action_tensor.item()
        
        # 2. 環境が1ステップ進む
        next_state, reward, done, _ = env.step(action)
        reward_tensor = torch.tensor([reward], device="cpu")
        
        # 3. 経験をリプレイバッファに保存
        agent.buffer.push(state, action_tensor, reward_tensor, next_state)
        
        state = next_state
        
        # 4. エージェントの学習
        agent.learn()
        
        # 5. ターゲットネットワークの更新
        if step % config.TARGET_UPDATE_FREQUENCY == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        rewards_log.append(reward)
        
        # 定期的に進捗を表示
        if step % 1000 == 0:
            avg_reward = np.mean(rewards_log[-1000:])
            print(f"ステップ: {step}/{config.SIMULATION_STEPS}, 平均報酬(直近1000): {avg_reward:.2f}")

    print("--- 学習終了 ---")

    # 学習が終わったら評価関数を呼び出す
    evaluate_agent(agent, NodeEnv(ConfigA()))


if __name__ == "__main__":
    train_dqn()