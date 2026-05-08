"""
耐受状态转移模型 (Tolerant State Transition Model)

模型描述：
- 初始状态：不健康状态 (1, 0, 0)
- 干预A (0,0,1)：连续 n 步可使状态转为健康 (0,0,0)
- 但如果在转换后继续使用干预A k 步，会产生耐受，状态回退到 (1,0,0)
- 产生耐受后，需要使用干预B (0,1,0) 连续 m 步才能完全转换到稳定健康状态

事件定义：
- E4a (完全转移): 先连续 n+k 步干预A，再连续 m 步干预B → 稳定健康 (0,0,0)
- E4b (耐受回退): 连续 n+k 步干预A（但未接干预B）→ 不健康 (1,0,0)（耐受）
- E4c (部分转移): 连续 n 步干预A（但不足 n+k）→ 健康 (0,0,0)（但不稳定）
- E4d (无有效转移): 其他情况 → 不健康 (1,0,0)
"""

from typing import List, Tuple, Optional, Dict, Any
from .base_model import BaseHealthModel, EventType


class TolerantStateTransitionModel(BaseHealthModel):
    """
    耐受状态转移模型
    
    参数:
        n: 部分转移所需连续干预A的步数
        k: 产生耐受所需的额外干预A步数（在n步之后）
        m: 耐受后完全转移所需连续干预B的步数
    """
    
    def __init__(self, n: int = 2, k: int = 2, m: int = 3):
        """
        初始化耐受状态转移模型
        
        :param n: 部分转移所需连续干预A步数
        :param k: 产生耐受所需的额外干预A步数
        :param m: 耐受后完全转移所需连续干预B步数
        """
        params = {
            "n": n,
            "k": k,
            "m": m,
            "type": "tolerant_state_transition"
        }
        super().__init__("TolerantStateTransitionModel", params)
        
        # 可观测状态
        self.UNHEALTHY = (1, 0, 0)      # 不健康状态（包括耐受状态）
        self.HEALTHY = (0, 0, 0)        # 健康状态
        
        # 干预定义
        self.INTERVENTION_A = (0, 0, 1)  # 干预A
        self.INTERVENTION_B = (0, 1, 0)  # 干预B
        self.NO_INTERVENTION = (0, 0, 0)  # 无干预
        
        # 内部状态跟踪
        self._consecutive_A_count = 0     # 连续干预A的计数
        self._consecutive_B_count = 0     # 连续干预B的计数
        self._is_in_partial_state = False # 是否处于部分转移后的健康状态
        self._is_tolerant = False         # 是否已产生耐受
    
    def _validate_params(self) -> None:
        """验证模型参数"""
        n = self.model_params["n"]
        k = self.model_params["k"]
        m = self.model_params["m"]
        
        assert n > 0, f"n ({n}) must be positive"
        assert k > 0, f"k ({k}) must be positive"
        assert m > 0, f"m ({m}) must be positive"
    
    def get_initial_state(self) -> Tuple:
        """获取初始状态：不健康 (1,0,0)"""
        return self.UNHEALTHY
    
    def get_available_interventions(self) -> List[Tuple]:
        """获取可用的干预动作"""
        return [self.INTERVENTION_A, self.INTERVENTION_B, self.NO_INTERVENTION]
    
    def check_event(
        self,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> EventType:
        """
        根据干预历史判断事件类型
        
        检查顺序：E4a -> E4b -> E4c -> E4d
        """
        t = len(intervention_history)
        n = self.model_params["n"]
        k = self.model_params["k"]
        m = self.model_params["m"]
        
        # 检查 E4a: 存在连续的 (n+k) 次干预A + m 次干预B
        required_length = n + k + m
        for i in range(t - required_length + 1):
            # 检查前 n+k 次是否是干预A
            a_part_valid = all(
                intervention_history[i + j] == self.INTERVENTION_A
                for j in range(n + k)
            )
            # 检查后 m 次是否是干预B
            b_part_valid = all(
                intervention_history[i + n + k + j] == self.INTERVENTION_B
                for j in range(m)
            )
            
            if a_part_valid and b_part_valid:
                return EventType.COMPLETE_TRANSITION
        
        # 检查 E4b: 最近 n+k 次都是干预A（耐受回退）
        if t >= n + k:
            if all(
                intervention_history[t - n - k + j] == self.INTERVENTION_A
                for j in range(n + k)
            ):
                return EventType.TOLERANCE_REVERSION
        
        # 检查 E4c: 最近 n 次都是干预A，但不足 n+k（部分转移）
        if t >= n:
            if all(
                intervention_history[t - n + j] == self.INTERVENTION_A
                for j in range(n)
            ):
                # 确保不是 E4b（即不足 n+k 次）
                if t < n + k or not all(
                    intervention_history[t - n - k + j] == self.INTERVENTION_A
                    for j in range(n + k)
                ):
                    return EventType.PARTIAL_TRANSITION
        
        # E4d: 其他情况
        return EventType.NO_EFFECTIVE_TRANSITION
    
    def get_next_state(
        self,
        current_state: Tuple,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> Tuple:
        """
        根据事件类型计算下一个状态
        
        转移规则：
        - E4a: 完全转移 → 稳定健康 (0,0,0)
        - E4b: 耐受回退 → 不健康 (1,0,0)（产生耐受）
        - E4c: 部分转移 → 健康 (0,0,0)（但不稳定）
        - E4d: 无有效转移 → 不健康 (1,0,0)
        """
        event = self.check_event(intervention_history, state_history)
        
        if event == EventType.COMPLETE_TRANSITION:
            # E4a: 完全转移，稳定健康
            self._consecutive_A_count = 0
            self._consecutive_B_count = 0
            self._is_in_partial_state = False
            self._is_tolerant = False
            return self.HEALTHY
        
        elif event == EventType.TOLERANCE_REVERSION:
            # E4b: 产生耐受，回到不健康
            self._consecutive_A_count = 0
            self._consecutive_B_count = 0
            self._is_in_partial_state = False
            self._is_tolerant = True
            return self.UNHEALTHY
        
        elif event == EventType.PARTIAL_TRANSITION:
            # E4c: 部分转移，进入健康状态（但不稳定）
            self._consecutive_A_count = 0
            self._consecutive_B_count = 0
            self._is_in_partial_state = True
            self._is_tolerant = False
            return self.HEALTHY
        
        else:  # NO_EFFECTIVE_TRANSITION
            # E4d: 无有效转移，不健康
            self._consecutive_A_count = 0
            self._consecutive_B_count = 0
            self._is_in_partial_state = False
            # 注意：耐受状态会持续，直到有有效的干预B序列
            return self.UNHEALTHY
    
    def reset_internal_state(self):
        """重置内部状态"""
        self._consecutive_A_count = 0
        self._consecutive_B_count = 0
        self._is_in_partial_state = False
        self._is_tolerant = False
    
    def simulate(
        self,
        intervention_sequence: List[Tuple],
        initial_state: Optional[Tuple] = None,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """模拟状态序列"""
        self.reset_internal_state()
        return super().simulate(intervention_sequence, initial_state, return_details)
    
    def get_internal_state(self) -> Dict[str, Any]:
        """获取内部状态（用于调试）"""
        return {
            "consecutive_A_count": self._consecutive_A_count,
            "consecutive_B_count": self._consecutive_B_count,
            "is_in_partial_state": self._is_in_partial_state,
            "is_tolerant": self._is_tolerant
        }


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    # 快速测试
    model = TolerantStateTransitionModel(n=2, k=2, m=3)
    
    print("耐受状态转移模型测试")
    print("=" * 60)
    print(f"参数: n={model.model_params['n']}, k={model.model_params['k']}, m={model.model_params['m']}")
    print(f"初始状态: {model.get_initial_state()}")
    print(f"可用干预: {model.get_available_interventions()}")