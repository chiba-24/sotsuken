""" 転送するパケットを選択する戦略 (Shortest TTL First) """
class ShortestTtlFirstStrategy:

    def select_packet(self, buffer):
        if not buffer:
            return None
        # バッファ内で最もTTLが小さいパケットを返す
        return min(buffer, key=lambda packet: packet.ttl)

""" 転送するパケットを選択する戦略 (First-In, First-Out) """
class FifoStrategy:
    def select_packet(self, buffer):
        if not buffer:
            return None
        # バッファの先頭にあるパケット（最も古くからあるもの）を返す
        return buffer[0]
    
