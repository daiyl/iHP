"""
拮抗状态转移模型 (Antagonistic State Transition Model)

模型描述：
- 初始状态：不健康状态 (1, 1, 1)
- 两种干预策略：
  - 干预A: a = (0, 0, 1)
  - 干预B: a = (0, 1, 0)
- 同时应用 (0,1,1) 会产生拮抗作用，状态不变
- 只有交替应用干预A和干预B，每种各 n 次，才能达到稳定健康 (0,0,0)

交替模式有两种：
1. 模式1: (0,0,1), (0,1,0), (0,0,1), (0,1,0), ... (以干预A开始)
2. 模式2: (0,1,0), (0,0,1), (0,1,0), (0,0,1), ... (以干预B开始)

事件定义：
- E7a (完全转移): 存在长度为 2n 的交替干预序列 → 健康 (0,0,0)
- E7b (无有效转移): 其他情况 → 不健康 (1,1,1)
"""

from typing import List, Tuple, Optional, Dict, Any
from .base_model import BaseHealthModel, EventType


class AntagonisticStateTransitionModel(BaseHealthModel):
    """
    拮抗状态转移模型
    
    参数:
        n: 每种干预需要应用的次数
    """
    
    def __init__(self, n: int = 3):
        """
        初始化拮抗状态转移模型
        
        :param n: 每种干预需要应用的次数（总步数为 2n）
        """
        params = {
            "n": n,
            "type": "antagonistic_state_transition"
        }
        super().__init__("AntagonisticStateTransitionModel", params)
        
        # 可观测状态
        self.UNHEALTHY = (1, 1, 1)    # 不健康状态
        self.HEALTHY = (0, 0, 0)      # 稳定健康状态
        
        # 干预定义
        self.INTERVENTION_A = (0, 0, 1)  # 干预A
        self.INTERVENTION_B = (0, 1, 0)  # 干预B
        self.INTERVENTION_AB = (0, 1, 1)  # 同时应用（拮抗）
        self.NO_INTERVENTION = (0, 0, 0)  # 无干预
        
        # 内部状态跟踪
        self._alternating_sequence_count = 0  # 当前交替序列长度
        self._expected_next = None            # 期望的下一个干预
        self._is_alternating = False          # 是否正在交替序列中
    
    def _validate_params(self) -> None:
        """验证模型参数"""
        n = self.model_params["n"]
        assert n > 0, f"n ({n}) must be positive"
    
    def get_initial_state(self) -> Tuple:
        """获取初始状态：不健康 (1,1,1)"""
        return self.UNHEALTHY
    
    def get_available_interventions(self) -> List[Tuple]:
        """获取可用的干预动作"""
        return [
            self.INTERVENTION_A, 
            self.INTERVENTION_B, 
            self.INTERVENTION_AB,
            self.NO_INTERVENTION
        ]
    
    def _is_alternating_pattern(
        self, 
        sequence: List[Tuple], 
        start_pattern: Tuple
    ) -> bool:
        """
        检查序列是否符合交替模式
        
        :param sequence: 干预序列
        :param start_pattern: 起始模式 (0,0,1) 或 (0,1,0)
        :return: 是否符合交替模式
        """
        if not sequence:
            return False
        
        # 确定交替模式
        if start_pattern == self.INTERVENTION_A:
            # 模式: A, B, A, B, ...
            for i, inter in enumerate(sequence):
                expected = self.INTERVENTION_A if i % 2 == 0 else self.INTERVENTION_B
                if inter != expected:
                    return False
        elif start_pattern == self.INTERVENTION_B:
            # 模式: B, A, B, A, ...
            for i, inter in enumerate(sequence):
                expected = self.INTERVENTION_B if i % 2 == 0 else self.INTERVENTION_A
                if inter != expected:
                    return False
        else:
            return False
        
        return True
    
    def check_event(
        self,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> EventType:
        """
        根据干预历史判断事件类型
        
        检查 E7a: 存在长度为 2n 的交替干预序列
        """
        t = len(intervention_history)
        n = self.model_params["n"]
        
        # 检查 E7a: 存在交替序列长度为 2n
        for i in range(t - 2 * n + 1):
            sub_sequence = intervention_history[i:i + 2 * n]
            
            # 检查模式1: 以干预A开始
            if self._is_alternating_pattern(sub_sequence, self.INTERVENTION_A):
                return EventType.COMPLETE_TRANSITION
            
            # 检查模式2: 以干预B开始
            if self._is_alternating_pattern(sub_sequence, self.INTERVENTION_B):
                return EventType.COMPLETE_TRANSITION
        
        # E7b: 其他情况
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
        - E7a: 完全转移 → 健康 (0,0,0)
        - E7b: 无有效转移 → 不健康 (1,1,1)
        """
        event = self.check_event(intervention_history, state_history)
        
        if event == EventType.COMPLETE_TRANSITION:
            # E7a: 完全转移，稳定健康
            self._alternating_sequence_count = 0
            self._expected_next = None
            self._is_alternating = False
            return self.HEALTHY
        
        else:  # NO_EFFECTIVE_TRANSITION
            # E7b: 无有效转移，不健康
            self._alternating_sequence_count = 0
            self._expected_next = None
            self._is_alternating = False
            return self.UNHEALTHY
    
    def reset_internal_state(self):
        """重置内部状态"""
        self._alternating_sequence_count = 0
        self._expected_next = None
        self._is_alternating = False
    
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
            "alternating_sequence_count": self._alternating_sequence_count,
            "expected_next": self._expected_next,
            "is_alternating": self._is_alternating
        }
    
    def check_alternating_progress(
        self,
        intervention_history: List[Tuple]
    ) -> Dict[str, Any]:
        """
        检查交替序列的进度（辅助方法）
        
        :param intervention_history: 干预历史
        :return: 包含最长交替序列信息的字典
        """
        t = len(intervention_history)
        n = self.model_params["n"]
        
        # 检查从结尾开始的最长交替序列
        longest_pattern = []
        
        # 尝试以干预A结尾的模式
        pattern_a = []
        for i in range(t - 1, -1, -1):
            expected = self.INTERVENTION_A if (t - 1 - i) % 2 == 0 else self.INTERVENTION_B
            if intervention_history[i] == expected:
                pattern_a.insert(0, intervention_history[i])
            else:
                break
        
        # 尝试以干预B结尾的模式
        pattern_b = []
        for i in range(t - 1, -1, -1):
            expected = self.INTERVENTION_B if (t - 1 - i) % 2 == 0 else self.INTERVENTION_A
            if intervention_history[i] == expected:
                pattern_b.insert(0, intervention_history[i])
            else:
                break
        
        # 选择更长的模式
        if len(pattern_a) >= len(pattern_b):
            longest_pattern = pattern_a
            pattern_type = "A开始"
        else:
            longest_pattern = pattern_b
            pattern_type = "B开始"
        
        return {
            "longest_alternating_length": len(longest_pattern),
            "pattern_type": pattern_type,
            "needed_for_complete": 2 * n - len(longest_pattern),
            "is_complete": len(longest_pattern) >= 2 * n
        }


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    # 快速测试
    model = AntagonisticStateTransitionModel(n=3)
    
    print("拮抗状态转移模型测试")
    print("=" * 60)
    print(f"参数: n={model.model_params['n']}")
    print(f"初始状态: {model.get_initial_state()}")
    print(f"可用干预: {model.get_available_interventions()}")