from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum


class EventType(Enum):
    """事件类型枚举"""
    COMPLETE_TRANSITION = "complete_transition"
    PARTIAL_TRANSITION = "partial_transition"
    PERIODIC_TO_UNHEALTHY = "periodic_to_unhealthy"
    PERIODIC_TO_HEALTHY = "periodic_to_healthy"
    STATE_MAINTENANCE = "state_maintenance"
    NO_EFFECTIVE_TRANSITION = "no_effective_transition"
	

class BaseHealthModel(ABC):
    """健康监护生成模型的统一基类"""
    
    def __init__(self, model_name: str, model_params: Dict[str, Any]):
        self.model_name = model_name
        self.model_params = model_params
        self._validate_params()
    
    @abstractmethod
    def _validate_params(self) -> None:
        pass
    
    @abstractmethod
    def get_initial_state(self) -> Tuple:
        pass
    
    @abstractmethod
    def get_available_interventions(self) -> List[Tuple]:
        pass
    
    @abstractmethod
    def check_event(
        self,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> EventType:
        pass
    
    @abstractmethod
    def get_next_state(
        self,
        current_state: Tuple,
        intervention_history: List[Tuple],
        state_history: Optional[List[Tuple]] = None
    ) -> Tuple:
        pass
    
    def simulate(
        self,
        intervention_sequence: List[Tuple],
        initial_state: Optional[Tuple] = None,
        return_details: bool = False
    ) -> Dict[str, Any]:
        if initial_state is None:
            state = self.get_initial_state()
        else:
            state = initial_state
        
        state_sequence = [state]
        event_sequence = []
        
        for t in range(1, len(intervention_sequence) + 1):
            history = intervention_sequence[:t]
            event = self.check_event(history, state_sequence[:-1])
            next_state = self.get_next_state(state, history, state_sequence[:-1])
            
            state_sequence.append(next_state)
            event_sequence.append(event)
            state = next_state
        
        result = {
            "states": state_sequence,
            "interventions": intervention_sequence
        }
        
        if return_details:
            result["events"] = event_sequence
            result["model_name"] = self.model_name
            result["model_params"] = self.model_params
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "initial_state": self.get_initial_state(),
            "available_interventions": self.get_available_interventions()
        }