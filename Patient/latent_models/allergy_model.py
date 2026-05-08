"""
过敏状态转移模型 (Allergy State Transition Model)

模型描述：
- 初始状态：不健康状态 (0, 0, 1)
- 干预A (1,1,0)：立即导致恶化，下一时间步转到不良反应状态 (1,1,0)
- 不良反应状态持续 k 步后，回到不健康状态 (0,0,1)
- 有效恢复需要连续 n 次干预B (1,0,0)，才能完全恢复到健康状态 (0,0,0)

事件定义：
- E6a (不良反应): 最近 k 步内存在干预A → 不良反应 (1,1,0)
- E6b (完全恢复): 存在连续 n 次干预B → 健康 (0,0,0)
- E6c (无有效转移): 其他情况 → 不健康 (0,0,1)
"""

from typing import List, Tuple, Optional, Dict, Any
from .base_model import BaseHealthModel, EventType


class AllergyStateTransitionModel(BaseHealthModel):
    """
    过敏状态转移模型
    
    参数:
        k: 不良反应状态持续的步数
        n: 完全恢复所需的连续干预B步数
    """
    
    def __init__(self, k: int = 2, n: int = 3):
        """
        初始化过敏状态转移模型
        
        :param k: 不良反应状态持续的步数
        :param n: 完全恢复所需的连续干预B步数
        """
        params = {
            "k": k,
            "n": n,
            "type": "allergy_state_transition"
        }
        super().__init__("AllergyStateTransitionModel", params)
        
        # 可观测状态
        self.UNHEALTHY = (0, 0, 1)    # 不健康状态
        self.HEALTHY = (0, 0, 0)      # 健康状态
        self.ADVERSE = (1, 1, 0)      # 不良反应状态
        
        # 干预定义
        self.INTERVENTION_A = (1, 1, 0)  # 干预A（引起过敏）
        self.INTERVENTION_B = (1, 0, 0)  # 干预B（治疗恢复）
        self.NO_INTERVENTION = (0, 0, 0)  # 无干预
        
        # 内部状态跟踪
        self._adverse_counter = 0         # 不良反应剩余步数
        self._consecutive_B_count = 0     # 连续干预B的计数
        self._is_in_adverse = False       # 是否处于不良反应状态
    
    def _validate_params(self) -> None:
        """验证模型参数"""
        k = self.model_params["k"]
        n = self.model_params["n"]
        
        assert k > 0, f"k ({k}) must be positive"
        assert n > 0, f"n ({n}) must be positive"
    
    def get_initial_state(self) -> Tuple:
        """获取初始状态：不健康 (0,0,1)"""
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
        
        检查顺序：E6a -> E6b -> E6c
        """
        t = len(intervention_history)
        k = self.model_params["k"]
        n = self.model_params["n"]
        
        # 检查 E6a: 最近 k 步内是否存在干预A
        # 注意：根据文档，只要在最近 k 步内有干预A，就会触发不良反应
        check_range = min(k, t)
        for i in range(check_range):
            if intervention_history[t - 1 - i] == self.INTERVENTION_A:
                return EventType.ADVERSE_REACTION
        
        # 检查 E6b: 存在连续 n 次干预B
        for i in range(t - n + 1):
            if all(
                intervention_history[i + j] == self.INTERVENTION_B
                for j in range(n)
            ):
                return EventType.COMPLETE_TRANSITION
        
        # E6c: 其他情况
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
        - E6a: 不良反应 → 不良反应状态 (1,1,0)
        - E6b: 完全恢复 → 健康 (0,0,0)
        - E6c: 无有效转移 → 不健康 (0,0,1)
        """
        event = self.check_event(intervention_history, state_history)
        
        if event == EventType.ADVERSE_REACTION:
            # E6a: 触发不良反应，设置持续计数器
            self._adverse_counter = self.model_params["k"]
            self._is_in_adverse = True
            self._consecutive_B_count = 0
            return self.ADVERSE
        
        elif event == EventType.COMPLETE_TRANSITION:
            # E6b: 完全恢复，进入稳定健康
            self._adverse_counter = 0
            self._is_in_adverse = False
            self._consecutive_B_count = 0
            return self.HEALTHY
        
        else:  # NO_EFFECTIVE_TRANSITION
            # E6c: 无有效转移
            self._adverse_counter = 0
            self._is_in_adverse = False
            self._consecutive_B_count = 0
            return self.UNHEALTHY
    
    def reset_internal_state(self):
        """重置内部状态"""
        self._adverse_counter = 0
        self._consecutive_B_count = 0
        self._is_in_adverse = False
    
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
            "adverse_counter": self._adverse_counter,
            "consecutive_B_count": self._consecutive_B_count,
            "is_in_adverse": self._is_in_adverse
        }


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    # 快速测试
    model = AllergyStateTransitionModel(k=2, n=3)
    
    print("过敏状态转移模型测试")
    print("=" * 60)
    print(f"参数: k={model.model_params['k']}, n={model.model_params['n']}")
    print(f"初始状态: {model.get_initial_state()}")
    print(f"可用干预: {model.get_available_interventions()}")