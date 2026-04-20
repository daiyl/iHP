"""
周期状态转移模型 (Periodic State Transition Model)

模型描述：
- 潜在健康状态在 unhealthy (0,1,0) 和 healthy (0,0,0) 之间周期性交替
- 每个状态持续 T 个时间步，周期长度为 2T
- 只有连续 n 次干预 a=(0,1,0) 才能将状态转换为稳定健康 (0,0,0)
- 否则，状态恢复其内在的周期性交替

事件定义：
- E2a (完全转移): 存在连续 n 次干预 → 稳定健康 (0,0,0)
- E2b (周期性转向不健康): 最近 T 步状态都是健康 → 下一个状态为不健康 (0,1,0)
- E2c (周期性转向健康): 最近 T 步状态都是不健康 → 下一个状态为健康 (0,0,0)
- E2d (状态维持): 其他情况 → 保持当前状态
"""

from typing import List, Tuple, Optional, Dict, Any
from .base_model import BaseHealthModel, EventType


class PeriodicStateTransitionModel(BaseHealthModel):
    """
    周期状态转移模型
    
    参数:
        T: 每个状态持续的时间步数
        n: 完全转移所需的连续干预步数
    """
    
    def __init__(self, T: int = 3, n: int = 2):
        """
        初始化周期状态转移模型
        
        :param T: 每个状态持续的时间步数
        :param n: 完全转移所需的连续干预步数
        """
        params = {
            "T": T,
            "n": n,
            "type": "periodic_state_transition"
        }
        super().__init__("PeriodicStateTransitionModel", params)
        
        # 可观测状态
        self.UNHEALTHY = (0, 1, 0)   # 不健康状态
        self.HEALTHY = (0, 0, 0)     # 健康状态
        
        # 干预定义
        self.INTERVENTION = (0, 1, 0)  # 干预
        self.NO_INTERVENTION = (0, 0, 0)  # 无干预
        
        # 内部状态跟踪
        self._period_counter = 0        # 当前周期内的位置
        self._current_phase = "healthy"  # 当前阶段: healthy 或 unhealthy
        self._stable_healthy = False    # 是否已进入稳定健康状态
    
    def _validate_params(self) -> None:
        """验证模型参数"""
        T = self.model_params["T"]
        n = self.model_params["n"]
        
        assert T > 0, f"T ({T}) must be positive"
        assert n > 0, f"n ({n}) must be positive"
    
    def get_initial_state(self) -> Tuple:
        """获取初始状态：不健康 (0,1,0)"""
        return self.UNHEALTHY
    
    def get_available_interventions(self) -> List[Tuple]:
        """获取可用的干预动作"""
        return [self.INTERVENTION, self.NO_INTERVENTION]
    
    def _get_periodic_next_state(self, current_state: Tuple) -> Tuple:
        """
        根据周期性规律计算下一个状态（不考虑干预）
        """
        if self._stable_healthy:
            return self.HEALTHY
        
        T = self.model_params["T"]
        
        # 更新周期计数器
        self._period_counter += 1
        
        # 检查是否需要切换阶段
        if self._period_counter >= T:
            self._period_counter = 0
            # 切换阶段
            if self._current_phase == "healthy":
                self._current_phase = "unhealthy"
                return self.UNHEALTHY
            else:
                self._current_phase = "healthy"
                return self.HEALTHY
        
        # 保持在当前阶段
        return self.HEALTHY if self._current_phase == "healthy" else self.UNHEALTHY
    
    def check_event(
        self,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> EventType:
        """
        根据干预历史和状态历史判断事件类型
        
        :param intervention_history: 历史干预序列
        :param state_history: 历史状态序列（用于检查周期性）
        """
        t = len(intervention_history)
        T = self.model_params["T"]
        n = self.model_params["n"]
        
        # 如果已经稳定健康，后续都是状态维持
        if self._stable_healthy:
            return EventType.STATE_MAINTENANCE
        
        # 检查 E2a: 存在连续 n 次干预
        for i in range(t - n + 1):
            if all(
                intervention_history[i + j] == self.INTERVENTION
                for j in range(n)
            ):
                return EventType.COMPLETE_TRANSITION
        
        # 检查 E2b 和 E2c 需要状态历史
        if state_history is not None and len(state_history) >= T:
            # 检查最近 T 步状态
            recent_states = state_history[-T:]
            
            # E2b: 最近 T 步都是健康
            if all(s == self.HEALTHY for s in recent_states):
                return EventType.PERIODIC_TO_UNHEALTHY
            
            # E2c: 最近 T 步都是不健康
            if all(s == self.UNHEALTHY for s in recent_states):
                return EventType.PERIODIC_TO_HEALTHY
        
        # E2d: 状态维持
        return EventType.STATE_MAINTENANCE
    
    def get_next_state(
        self,
        current_state: Tuple,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> Tuple:
        """
        根据事件类型计算下一个状态
        
        转移规则：
        - E2a: 完全转移 → 稳定健康 (0,0,0)
        - E2b: 周期性转向不健康 → 不健康 (0,1,0)
        - E2c: 周期性转向健康 → 健康 (0,0,0)
        - E2d: 状态维持 → 保持当前状态
        """
        event = self.check_event(intervention_history, state_history)
        
        if event == EventType.COMPLETE_TRANSITION:
            # E2a: 完全转移到稳定健康
            self._stable_healthy = True
            self._period_counter = 0
            self._current_phase = "healthy"
            return self.HEALTHY
        
        elif event == EventType.PERIODIC_TO_UNHEALTHY:
            # E2b: 周期性转向不健康
            # 重置周期计数器，开始不健康阶段
            self._period_counter = 1
            self._current_phase = "unhealthy"
            return self.UNHEALTHY
        
        elif event == EventType.PERIODIC_TO_HEALTHY:
            # E2c: 周期性转向健康
            # 重置周期计数器，开始健康阶段
            self._period_counter = 1
            self._current_phase = "healthy"
            return self.HEALTHY
        
        else:  # STATE_MAINTENANCE
            # E2d: 状态维持
            if self._stable_healthy:
                return self.HEALTHY
            
            # 根据周期性规律更新状态
            return self._get_periodic_next_state(current_state)
    
    def reset_internal_state(self):
        """重置内部状态"""
        self._period_counter = 0
        self._current_phase = "unhealthy"  # 初始为不健康
        self._stable_healthy = False
    
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
            "period_counter": self._period_counter,
            "current_phase": self._current_phase,
            "stable_healthy": self._stable_healthy
        }


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    # 快速测试
    model = PeriodicStateTransitionModel(T=3, n=2)
    
    print("周期状态转移模型测试")
    print("=" * 60)
    print(f"参数: T={model.model_params['T']}, n={model.model_params['n']}")
    print(f"初始状态: {model.get_initial_state()}")
    print(f"可用干预: {model.get_available_interventions()}")