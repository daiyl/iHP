from typing import List, Tuple, Dict, Any
import json


def format_state_sequence(
    states: List[Tuple],
    interventions: List[Tuple],
    events: List = None
) -> str:
    """格式化输出状态序列"""
    output = []
    output.append(f"{'Step':<6} {'Intervention':<20} {'State':<20}")
    if events:
        output.append(f"{'Event':<20}")
    output.append("-" * 60)
    
    for i in range(len(states)):
        if i == 0:
            inter_str = "initial"
        else:
            inter_str = str(interventions[i-1])
        
        state_str = str(states[i])
        
        if events and i > 0:
            event_str = events[i-1].value if hasattr(events[i-1], 'value') else str(events[i-1])
            output.append(f"{i:<6} {inter_str:<20} {state_str:<20} {event_str:<20}")
        else:
            output.append(f"{i:<6} {inter_str:<20} {state_str:<20}")
    
    return "\n".join(output)

def save_simulation_result(result: Dict[str, Any], filepath: str) -> None:
    """保存模拟结果到JSON文件"""
    serializable_result = {
        "states": [list(s) if isinstance(s, tuple) else s for s in result["states"]],
        "interventions": [list(i) if isinstance(i, tuple) else i for i in result["interventions"]]
    }
    
    if "events" in result:
        serializable_result["events"] = [e.value if hasattr(e, 'value') else str(e) for e in result["events"]]
    
    if "model_name" in result:
        serializable_result["model_name"] = result["model_name"]
        serializable_result["model_params"] = result["model_params"]
    
    with open(filepath, 'w') as f:
        json.dump(serializable_result, f, indent=2)