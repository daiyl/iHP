from typing import List, Tuple, Optional, Dict, Any
from models.base_model import BaseHealthModel, EventType


class DelayedStateTransitionModel(BaseHealthModel):
    """
    延迟状态转移模型
    
    可观测状态说明：
    - 健康状态 HEALTHY = (0, 0, 0)
    - 不健康状态 UNHEALTHY = (0, 0, 1)
    
    状态转移逻辑：
    - 完全转移 (E1a): 连续 m 步干预 → 稳定健康状态
    - 部分转移 (E1b): 连续 n 步干预 → 进入中间状态，观测为健康 (0,0,0)
    - 无有效转移 (E1c): 其他情况 → 不健康状态 (0,0,1)
    
    注意：中间状态是不可观测的潜在状态，但观测到的状态是健康 (0,0,0)
    """
    
    def __init__(self, n: int = 2, m: int = 4, k: int = 2):
        params = {
            "n": n,
            "m": m,
            "k": k,
            "type": "delayed_state_transition"
        }
        super().__init__("DelayedStateTransitionModel", params)
        
        # 可观测状态
        self.UNHEALTHY = (0, 0, 1)
        self.HEALTHY = (0, 0, 0)
        
        # 干预定义
        self.INTERVENTION = (0, 0, 1)
        self.NO_INTERVENTION = (0, 0, 0)
        
        # 内部状态跟踪（用于模拟不可观测的中间状态）
        self._partial_transition_counter = 0  # 记录部分转移后剩余的中间状态步数
    
    def _validate_params(self) -> None:
        n = self.model_params["n"]
        m = self.model_params["m"]
        k = self.model_params["k"]
        
        assert m > n, f"m ({m}) must be greater than n ({n})"
        assert n > 0, f"n ({n}) must be positive"
        assert m > 0, f"m ({m}) must be positive"
        assert k >= 0, f"k ({k}) must be non-negative"
    
    def get_initial_state(self) -> Tuple:
        """获取初始状态：不健康"""
        return self.UNHEALTHY
    
    def get_available_interventions(self) -> List[Tuple]:
        """获取可用的干预动作"""
        return [self.INTERVENTION, self.NO_INTERVENTION]
    
    def check_event(
        self,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> EventType:
        t = len(intervention_history)
        n = self.model_params["n"]
        m = self.model_params["m"]
        k = self.model_params["k"]
        
        # 检查 E1a: 存在连续 m 次干预
        for i in range(t - m + 1):
            if all(
                intervention_history[i + j] == self.INTERVENTION
                for j in range(m)
            ):
                return EventType.COMPLETE_TRANSITION
        
        # 检查 E1b: 存在连续 n 次干预且最近 k 步内
        for j in range(min(k, t - n + 1)):
            start = t - n - j
            if start < 0:
                continue
            if all(
                intervention_history[start + x] == self.INTERVENTION
                for x in range(n)
            ):
                return EventType.PARTIAL_TRANSITION
        
        return EventType.NO_EFFECTIVE_TRANSITION
    
    def get_next_state(
        self,
        current_state: Tuple,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> Tuple:
        """
        根据当前状态和历史计算下一个可观测状态
        
        转移规则：
        - E1a (完全转移): 返回健康状态 (0,0,0)，且稳定
        - E1b (部分转移): 返回健康状态 (0,0,0)，但内部处于中间状态
        - E1c (无有效转移): 返回不健康状态 (0,0,1)
        """
        event = self.check_event(intervention_history)
        
        if event == EventType.COMPLETE_TRANSITION:
            # 完全转移到健康状态，清除中间状态计数器
            self._partial_transition_counter = 0
            return self.HEALTHY
        
        elif event == EventType.PARTIAL_TRANSITION:
            # 部分转移：进入中间状态，设置计数器，观测状态为健康
            self._partial_transition_counter = self.model_params["k"]
            return self.HEALTHY
        
        else:  # NO_EFFECTIVE_TRANSITION
            # 无有效干预；不在中间状态，返回不健康
                return self.UNHEALTHY
    
    def reset_internal_state(self):
        """重置内部状态计数器（用于新的模拟）"""
        self._partial_transition_counter = 0
    
    def simulate(
        self,
        intervention_sequence: List[Tuple],
        initial_state: Optional[Tuple] = None,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """重写simulate方法以重置内部状态"""
        self.reset_internal_state()
        return super().simulate(intervention_sequence, initial_state, return_details)
    
    def get_internal_state(self) -> Dict[str, Any]:
        """获取内部状态（用于调试）"""
        return {
            "partial_transition_counter": self._partial_transition_counter,
            "is_in_intermediate_state": self._partial_transition_counter > 0
        }