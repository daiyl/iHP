"""
协同状态转移模型 (Synergistic State Transition Model)

模型描述：
- 初始状态：不健康状态 (0, 1, 1)
- 单一干预 (0,0,1)：连续 n 步可使状态转为稳定健康 (0,0,0)
- 联合干预 (0,1,1)：加速恢复过程，连续 m 步即可转为稳定健康 (m < n)

事件定义：
- E3a (单一干预转移): 存在连续 n 次单一干预 (0,0,1) → 稳定健康 (0,0,0)
- E3b (协同转移): 存在连续 m 次联合干预 (0,1,1) → 稳定健康 (0,0,0)
- E3c (无有效转移): 其他情况 → 不健康 (0,1,1)
"""

from typing import List, Tuple, Optional, Dict, Any
from .base_model import BaseHealthModel, EventType


class SynergisticStateTransitionModel(BaseHealthModel):
    """
    协同状态转移模型
    
    参数:
        n: 单一干预所需连续步数
        m: 联合干预所需连续步数 (m < n)
    """
    
    def __init__(self, n: int = 5, m: int = 3):
        """
        初始化协同状态转移模型
        
        :param n: 单一干预所需连续步数
        :param m: 联合干预所需连续步数
        """
        params = {
            "n": n,
            "m": m,
            "type": "synergistic_state_transition"
        }
        super().__init__("SynergisticStateTransitionModel", params)
        
        # 可观测状态
        self.UNHEALTHY = (0, 1, 1)    # 不健康状态
        self.HEALTHY = (0, 0, 0)      # 稳定健康状态
        
        # 干预定义
        self.SINGLE_INTERVENTION = (0, 0, 1)   # 单一干预
        self.JOINT_INTERVENTION = (0, 1, 1)    # 联合干预
        self.NO_INTERVENTION = (0, 0, 0)       # 无干预
        
        # 内部状态跟踪
        self._consecutive_single_count = 0   # 连续单一干预计数
        self._consecutive_joint_count = 0    # 连续联合干预计数
    
    def _validate_params(self) -> None:
        """验证模型参数"""
        n = self.model_params["n"]
        m = self.model_params["m"]
        
        assert m < n, f"m ({m}) must be less than n ({n})"
        assert n > 0, f"n ({n}) must be positive"
        assert m > 0, f"m ({m}) must be positive"
    
    def get_initial_state(self) -> Tuple:
        """获取初始状态：不健康 (0,1,1)"""
        return self.UNHEALTHY
    
    def get_available_interventions(self) -> List[Tuple]:
        """获取可用的干预动作"""
        return [self.SINGLE_INTERVENTION, self.JOINT_INTERVENTION, self.NO_INTERVENTION]
    
    def _update_consecutive_counts(
        self,
        intervention_history: List[Tuple]
    ) -> None:
        """
        更新连续干预计数（基于最近的历史）
        """
        if not intervention_history:
            self._consecutive_single_count = 0
            self._consecutive_joint_count = 0
            return
        
        # 获取最近的干预
        last_intervention = intervention_history[-1]
        
        if last_intervention == self.SINGLE_INTERVENTION:
            # 如果上一个也是单一干预，增加计数；否则重置
            if len(intervention_history) >= 2 and intervention_history[-2] == self.SINGLE_INTERVENTION:
                self._consecutive_single_count += 1
            else:
                self._consecutive_single_count = 1
            self._consecutive_joint_count = 0
        
        elif last_intervention == self.JOINT_INTERVENTION:
            # 如果上一个也是联合干预，增加计数；否则重置
            if len(intervention_history) >= 2 and intervention_history[-2] == self.JOINT_INTERVENTION:
                self._consecutive_joint_count += 1
            else:
                self._consecutive_joint_count = 1
            self._consecutive_single_count = 0
        
        else:  # 无干预
            self._consecutive_single_count = 0
            self._consecutive_joint_count = 0
    
    def check_event(
        self,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> EventType:
        """
        根据干预历史判断事件类型
        
        :param intervention_history: 历史干预序列
        :param state_history: 历史状态序列（本模型中未使用）
        """
        t = len(intervention_history)
        n = self.model_params["n"]
        m = self.model_params["m"]
        
        # 检查 E3a: 存在连续 n 次单一干预 (0,0,1)
        for i in range(t - n + 1):
            if all(
                intervention_history[i + j] == self.SINGLE_INTERVENTION
                for j in range(n)
            ):
                return EventType.COMPLETE_TRANSITION
        
        # 检查 E3b: 存在连续 m 次联合干预 (0,1,1)
        for i in range(t - m + 1):
            if all(
                intervention_history[i + j] == self.JOINT_INTERVENTION
                for j in range(m)
            ):
                return EventType.SYNERGISTIC_TRANSITION
        
        # E3c: 无有效转移
        return EventType.NO_EFFECTIVE_TRANSITION
    
    def get_next_state(
        self,
        current_state: Tuple,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> Tuple:
        """
        根据事件类型计算下一个状态
        
        转移规则（根据文档）：
        - E3a: 单一干预转移 → 稳定健康 (0,0,0)
        - E3b: 协同转移 → 稳定健康 (0,0,0)
        - E3c: 无有效转移 → 不健康 (0,1,1)
        """
        event = self.check_event(intervention_history, state_history)
        
        if event == EventType.COMPLETE_TRANSITION:
            # E3a: 单一干预达到 n 次，转移到稳定健康
            self._consecutive_single_count = 0
            self._consecutive_joint_count = 0
            return self.HEALTHY
        
        elif event == EventType.SYNERGISTIC_TRANSITION:
            # E3b: 联合干预达到 m 次，加速转移到稳定健康
            self._consecutive_single_count = 0
            self._consecutive_joint_count = 0
            return self.HEALTHY
        
        else:  # NO_EFFECTIVE_TRANSITION
            # E3c: 无有效转移，返回不健康状态 (0,1,1)
            self._consecutive_single_count = 0
            self._consecutive_joint_count = 0
            return self.UNHEALTHY
    
    def reset_internal_state(self):
        """重置内部状态"""
        self._consecutive_single_count = 0
        self._consecutive_joint_count = 0
    
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
            "consecutive_single_count": self._consecutive_single_count,
            "consecutive_joint_count": self._consecutive_joint_count
        }


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    # 快速测试
    model = SynergisticStateTransitionModel(n=5, m=3)
    
    print("协同状态转移模型测试")
    print("=" * 60)
    print(f"参数: n={model.model_params['n']}, m={model.model_params['m']}")
    print(f"初始状态: {model.get_initial_state()}")
    print(f"可用干预: {model.get_available_interventions()}")
    
    # 简单测试序列
    print("\n测试: 3次联合干预 (m=3)")
    interventions = [(0, 1, 1)] * 3
    result = model.simulate(interventions)
    
    print(f"干预序列: {interventions}")
    print(f"状态序列: {result['states']}")