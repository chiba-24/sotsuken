from abc import ABC, abstractmethod
# environmentsフォルダのbase_envからBaseEnvをインポートして型ヒントに使う
from environments.base_env import BaseEnv 

class BaseStrategy(ABC):
    """
    全ての転送戦略クラスが継承すべき、基本となる設計図（抽象基底クラス）。
    """

    @abstractmethod
    def __init__(self, config):
        """
        戦略を初期化する。
        Args:
            config: 実験設定オブジェクト
        """
        self.config = config

    @abstractmethod
    def select_action(self, env: BaseEnv):
        """
        現在の環境の状態(env)を観測し、次に取るべき行動(action)を決定して返す。
        
        Args:
            env (BaseEnv): 現在の環境オブジェクト
        
        Returns:
            action: 選択された行動（転送するパケットのインデックスなど）
        """
        pass

    # 実装が必須ではない
    def train(self, env: BaseEnv):
        """
        学習が必要な戦略（DQNなど）のための学習メソッド。
        学習が不要な戦略（FIFOなど）では、このメソッドを実装する必要はない。
        """
        # デフォルトでは何もしない
        pass