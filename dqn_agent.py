import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# エージェントが経験した「状態，行動．報酬，次の報酬」という一連の出来事をまとめて保存
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state'))

# エージェントの経験を蓄積して，学習時にランダムに経験を取り出す．
class ReplayBuffer:
    # 最初に呼ばれる関数
    # capacityで指定された最大容量を持つ記憶領域を作成．容量を超えると古い経験から自動的に削除．
    ## 削除順の制御も必要かも?
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    # 新しい経験を受け取り，Experienceの形に変換して，memoryの末尾に追加．
    def push(self, *args):
        self.memory.append(Experience(*args))

    # memoryに蓄積された経験の中から，batch_sizeで指定された数だけランダムに選んで返す．
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    # バッファに保存されている経験の数を返す．
    def __len__(self):
        return len(self.memory)

# Q値を予測するためのNN本体
class QNetwork(nn.Module):
    # ネットワークの構造を定義
    def __init__(self, state_size, action_size, hidden_sizes):
        # 親クラスの「nn.Module」も継承に必要
        super(QNetwork, self).__init__()

        layers = []
        input_size = state_size
        # hidden_sizesリストに基づいて中間層を動的に構築
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size)) # 線形層を追加
            layers.append(nn.ReLU()) # 活性化関数ReLUを追加
            input_size = hidden_size # 次の層の入力サイズを更新

        # 最終的な出力層を追加
        layers.append(nn.Linear(input_size, action_size))

        # layersリストの中に順番に格納した層（nn.Linear, nn.ReLUなど）を
        # 1つの連続したネットワークモジュールにまとめる
        # *layers: [層1, 層2, ...] というリストを，層1, 層2, ... のように個々の要素に展開
        self.network = nn.Sequential(*layers)

        # 入力データxをself.networkに渡すだけで，nn.Sequentialが自動的にリストの先頭から順番に層を適用して最終的な計算結果を返す
        def forward(self, x):
            return self.network(x)

    #     # 結合層を定義．
    #     # nn.Linear：データを分析して重みを学習する線形層．
    #     # 入力 -> 中間層1 -> 中間層2 -> 出力層(各行動のQ値)
    #     self.layer1 = nn.Linear(state_size, 128)
    #     self.layer2 = nn.Linear(128, 128)
    #     self.layer3 = nn.Linear(128, action_size)

    # # 状態xのデータがどのように流れるかを定義
    # # xには「正規化されたTTL」と「正規化されたサイズ」を並べた1次元配列
    # def forward(self, x):
    #     # 活性化関数ReLUを適用して表現力向上
    #     # F.relu：曲線的な関係をモデル化する能力を付与する活性化関数．
    #     x = F.relu(self.layer1(x))
    #     x = F.relu(self.layer2(x))
    #     return self.layer3(x)

# DQNアルゴリズム全体を管理して，行動決定や学習を行うエージェント本体
class DqnAgent:
    # エージェントが必要とする情報の初期化
    def __init__(self, state_size, action_size, config):
        # 状態の次元数の初期化
        self.state_size = state_size
        # 行動の次元数の初期化
        self.action_size = action_size
        # ネットワーク設定の初期化
        self.config = config
        # ステップ数カウンターの初期化
        self.steps_done = 0

        # QNetworkの初期化時に、configから隠れ層のサイズリストを渡す
        # policy_net：実際に行動を決定し、学習で更新されるメインのネットワーク
        self.policy_net = QNetwork(state_size, action_size, config.HIDDEN_LAYER_SIZES).to(device)
        # target_net：学習を安定させるために、学習目標（TDターゲット）の計算に使うネットワーク
        self.target_net = QNetwork(state_size, action_size, config.HIDDEN_LAYER_SIZES).to(device)

        # target_netの重みをpolicy_netと全く同じ状態に初期化
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # target_netを学習しない評価モードに設定
        self.target_net.eval()

        # policy_netの重みを更新するための最適化アルゴリズム（Adam）を設定
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        # ReplayBufferの初期化時に、configから容量を渡す
        self.buffer = ReplayBuffer(config.REPLAY_BUFFER_CAPACITY)

    def select_action(self, state):
        epsilon = self.config.EPSILON_END + (self.config.EPSILON_START - self.config.EPSILON_END) * \
                  np.exp(-1. * self.steps_done / self.config.EPSILON_DECAY)
        self.steps_done += 1

        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

    def learn(self):
        if len(self.buffer) < self.config.BATCH_SIZE:
            return
        
        experiences = self.buffer.sample(self.config.BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        state_batch = torch.cat([torch.tensor(s, device=device).unsqueeze(0) for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat([torch.tensor(s, device=device).unsqueeze(0) for s in batch.next_state])
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.config.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()