from abc import ABC, abstractmethod     # 抽象基底クラスを作るための道具をインポート

# ABCを継承することで、このクラスが抽象基底クラスであることを明示
class BaseEnv(ABC):
    """
    全てのシミュレーション環境クラスが継承すべき、基本となる設計図（抽象基底クラス）。
    このクラスを継承する子クラスは、@abstractmethodが付いた全てのメソッドを
    実装することが強制される。
    """

    # 「@abstractmethod」が付いたメソッドは「実装が必須のメソッド」であることを明示
    # 中身はpassでOK
    @abstractmethod
    def __init__(self, config):
        """
        環境を初期化する。
        Args:
            config: 実験設定オブジェクト
        """
        pass

    @abstractmethod
    def reset(self):
        """
        環境を初期状態にリセットする。
        Returns:
            最初の状態(state)
        """
        pass

    @abstractmethod
    def update_time(self, current_step):
        """
        時間が1ステップ進んだ際の、環境の自動的な変化を処理する。
        （例: パケット到着、TTL減少、リンク容量の変動など）
        
        Args:
            current_step (int): 現在のシミュレーションステップ数
        
        Returns:
            (float, dict): 時間経過による報酬, 時間経過に関する統計情報
        """
        pass

    @abstractmethod
    def transmit_packet(self, action):
        """
        エージェントから受け取ったaction（どのパケットを転送するか）を処理する。
        
        Args:
            action: エージェントが選択した行動
        
        Returns:
            (float, int, bool): 行動による報酬, 転送数, 転送が成功したか
        """
        pass
    
    @abstractmethod
    def get_state(self):
        """
        現在の環境の状態を、エージェントが理解できる形式で返す。
        """
        pass