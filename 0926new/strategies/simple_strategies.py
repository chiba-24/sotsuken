from .base_strategy import BaseStrategy

class FifoStrategy(BaseStrategy):
    """
    転送戦略: FIFO (First-In, First-Out)
    バッファに最も古くから存在するパケット（キューの先頭）を常に選択する。
    """
    
    def __init__(self, config):
        # 親クラスの初期化処理を呼び出す
        super().__init__(config)

    def select_action(self, env):
        """
        次に取るべき行動(action)を決定して返す。
        """
        # 1. バッファが空かどうかをチェック
        if not env.buffer:
            # 送るべきパケットがないので、何もしない (Noneを返す)
            return None
        
        # 2. バッファの先頭（インデックス0）をactionとして選択
        # これが最も古くからキューイングされているパケットになる
        return 0

class ShortestTtlFirstStrategy(BaseStrategy):
    """
    転送戦略: 最小TTL優先 (Shortest TTL First)
    バッファ内で最もTTLが小さいパケットを選択する。
    """

    def __init__(self, config):
        super().__init__(config)

    def select_action(self, env):
        """
        次に取るべき行動(action)を決定して返す。
        """
        # 1. バッファが空かどうかをチェック
        if not env.buffer:
            return None
            
        # 2. バッファ内の全パケットを調べて、最もTTLが小さいパケットを見つける
        min_ttl_packet = min(env.buffer, key=lambda packet: packet.ttl)
        
        # 3. 見つけたパケットのバッファ内でのインデックス(何番目か)を取得してactionとして返す
        try:
            action = list(env.buffer).index(min_ttl_packet)
            return action
        except ValueError:
            # 非常に稀だが、探索中にパケットが消えた場合
            return None