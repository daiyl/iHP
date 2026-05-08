"""
依赖诱导恶化状态转移模型 (Dependency-Induced Deterioration State Transition Model)

模型描述：
- 初始状态：不健康状态 (0, 0, 1)
- 干预 (1,0,1)：连续 n 步可使状态转为稳定健康 (0,0,0)
- 如果在任何时候停止干预 (0,0,0)，状态会在下一步恶化到 (1,0,1)
- 恶化后，需要重新连续 n 次干预才能恢复到健康状态

事件定义：
- E5a (稳定健康): 最近 n 步都是干预 (1,0,1) → 健康 (0,0,0)
- E5b (恶化): 存在连续 n 次干预后停止干预 → 恶化到 (1,0,1)
- E5c (无有效转移): 其他情况 → 不健康 (0,0,1)
"""

from typing import List, Tuple, Optional, Dict, Any
from .base_model import BaseHealthModel, EventType


class DependencyDeteriorationModel(BaseHealthModel):
    """
    依赖诱导恶化状态转移模型
    
    参数:
        n: 达到健康所需的连续干预步数
    """
    
    def __init__(self, n: int = 3):
        """
        初始化依赖诱导恶化状态转移模型
        
        :param n: 达到健康所需的连续干预步数
        """
        params = {
            "n": n,
            "type": "dependency_deterioration"
        }
        super().__init__("DependencyDeteriorationModel", params)
        
        # 可观测状态
        self.UNHEALTHY = (0, 0, 1)    # 不健康状态
        self.HEALTHY = (0, 0, 0)      # 稳定健康状态
        self.DETERIORATED = (1, 0, 1)  # 恶化状态
        
        # 干预定义
        self.INTERVENTION = (1, 0, 1)  # 干预
        self.NO_INTERVENTION = (0, 0, 0)  # 无干预
        
        # 内部状态跟踪
        self._consecutive_intervention_count = 0  # 连续干预计数
        self._is_healthy = False                  # 是否处于健康状态
        self._is_deteriorated = False             # 是否处于恶化状态
    
    def _validate_params(self) -> None:
        """验证模型参数"""
        n = self.model_params["n"]
        assert n > 0, f"n ({n}) must be positive"
    
    def get_initial_state(self) -> Tuple:
        """获取初始状态：不健康 (0,0,1)"""
        return self.UNHEALTHY
    
    def get_available_interventions(self) -> List[Tuple]:
        """获取可用的干预动作"""
        return [self.INTERVENTION, self.NO_INTERVENTION]
    
    def _update_consecutive_count(
        self,
        intervention_history: List[Tuple]
    ) -> None:
        """
        更新连续干预计数
        """
        if not intervention_history:
            self._consecutive_intervention_count = 0
            return
        
        # 获取最近的干预
        last_intervention = intervention_history[-1]
        
        if last_intervention == self.INTERVENTION:
            # 如果上一个也是干预，增加计数；否则重置
            if len(intervention_history) >= 2 and intervention_history[-2] == self.INTERVENTION:
                self._consecutive_intervention_count += 1
            else:
                self._consecutive_intervention_count = 1
        else:
            # 无干预，重置计数
            self._consecutive_intervention_count = 0
    
    def check_event(
        self,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> EventType:
        """
        根据干预历史判断事件类型
        
        检查顺序：E5a -> E5b -> E5c
        """
        t = len(intervention_history)
        n = self.model_params["n"]
        
        # 更新连续干预计数
        self._update_consecutive_count(intervention_history)
        
        # 检查 E5a: 最近 n 步都是干预 (1,0,1)
        if t >= n:
            if all(
                intervention_history[t - n + j] == self.INTERVENTION
                for j in range(n)
            ):
                return EventType.COMPLETE_TRANSITION
        
        # 检查 E5b: 存在连续 n 次干预后立即停止干预
        # 条件：存在 i 使得 a_i...a_{i+n-1} 都是干预，且 a_{i+n} 是无干预
        for i in range(t - n):
            # 检查连续 n 次干预
            if all(
                intervention_history[i + j] == self.INTERVENTION
                for j in range(n)
            ):
                # 检查接下来的第 n 步是否是无干预
                if i + n < t and intervention_history[i + n] == self.NO_INTERVENTION:
                    return EventType.DETERIORATION
        
        # E5c: 其他情况
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
        - E5a: 稳定健康 → 健康 (0,0,0)
        - E5b: 恶化 → 恶化状态 (1,0,1)
        - E5c: 无有效转移 → 不健康 (0,0,1)
        """
        event = self.check_event(intervention_history, state_history)
        
        if event == EventType.COMPLETE_TRANSITION:
            # E5a: 稳定健康
            self._consecutive_intervention_count = 0
            self._is_healthy = True
            self._is_deteriorated = False
            return self.HEALTHY
        
        elif event == EventType.DETERIORATION:
            # E5b: 恶化
            self._consecutive_intervention_count = 0
            self._is_healthy = False
            self._is_deteriorated = True
            return self.DETERIORATED
        
        else:  # NO_EFFECTIVE_TRANSITION
            # E5c: 无有效转移，不健康
            self._consecutive_intervention_count = 0
            self._is_healthy = False
            self._is_deteriorated = False
            return self.UNHEALTHY
    
    def reset_internal_state(self):
        """重置内部状态"""
        self._consecutive_intervention_count = 0
        self._is_healthy = False
        self._is_deteriorated = False
    
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
            "consecutive_intervention_count": self._consecutive_intervention_count,
            "is_healthy": self._is_healthy,
            "is_deteriorated": self._is_deteriorated
        }


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    # 快速测试
    model = DependencyDeteriorationModel(n=3)
    
    print("依赖诱导恶化状态转移模型测试")
    print("=" * 60)
    print(f"参数: n={model.model_params['n']}")
    print(f"初始状态: {model.get_initial_state()}")
    print(f"可用干预: {model.get_available_interventions()}")