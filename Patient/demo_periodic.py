"""
周期状态转移模型演示
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.periodic_model import PeriodicStateTransitionModel
from models.base_model import EventType


def demo_periodic_model():
    """演示周期状态转移模型"""
    print("=" * 80)
    print("周期状态转移模型演示")
    print("=" * 80)
    
    # 创建模型实例
    model = PeriodicStateTransitionModel(T=3, n=2)
    
    print(f"\n📊 模型信息:")
    print(f"   参数: T={model.model_params['T']}, n={model.model_params['n']}")
    print(f"   初始状态: (0,1,0) 不健康")
    print(f"   周期长度: {2*model.model_params['T']} 步")
    print(f"   可用干预: (0,1,0) 🔵 干预, (0,0,0) ⚪ 无干预")
    
    print(f"\n📖 状态说明:")
    print(f"   (0,1,0) → 不健康状态")
    print(f"   (0,0,0) → 健康状态")
    
    print(f"\n📖 事件说明:")
    print(f"   E2a (完全转移): 连续 {model.model_params['n']} 次干预 → 稳定健康")
    print(f"   E2b (周期性转向不健康): 最近 T 步都是健康 → 下一个为不健康")
    print(f"   E2c (周期性转向健康): 最近 T 步都是不健康 → 下一个为健康")
    print(f"   E2d (状态维持): 其他情况 → 保持当前状态")
    
    # 场景1: 无干预的自然周期性
    print("\n" + "=" * 80)
    print("场景1: 无干预 - 自然周期性交替")
    print("=" * 80)
    
    interventions_scene1 = [(0, 0, 0)] * 12  # 12步无干预
    
    result1 = model.simulate(interventions_scene1, return_details=True)
    _print_results(result1, model, title="自然周期 (T=3: 3步不健康 → 3步健康 → 循环)")
    
    # 场景2: 部分干预（未达到完全转移）
    print("\n" + "=" * 80)
    print("场景2: 部分干预 - 中断周期性但未完全转移")
    print("=" * 80)
    
    interventions_scene2 = [
        (0, 0, 0),  # t=1: 无干预
        (0, 0, 0),  # t=2: 无干预
        (0, 1, 0),  # t=3: 干预（1次）
        (0, 1, 0),  # t=4: 干预（1次，未达到n=2）
        (0, 0, 0),  # t=5: 无干预
        (0, 0, 0),  # t=6: 无干预
        (0, 0, 0),  # t=7: 无干预
    ]
    
    result2 = model.simulate(interventions_scene2, return_details=True)
    _print_results(result2, model, title="1次干预不足以触发完全转移")
    
    # 场景3: 完全转移
    print("\n" + "=" * 80)
    print("场景3: 完全转移 - 连续2次干预")
    print("=" * 80)
    
    interventions_scene3 = [
        (0, 0, 0),  # t=1: 无干预
        (0, 0, 0),  # t=2: 无干预
        (0, 1, 0),  # t=3: 干预
        (0, 1, 0),  # t=4: 干预（连续2次，触发E2a）
        (0, 0, 0),  # t=5: 无干预
        (0, 0, 0),  # t=6: 无干预
        (0, 0, 0),  # t=7: 无干预
    ]
    
    result3 = model.simulate(interventions_scene3, return_details=True)
    _print_results(result3, model, title="连续2次干预 → 稳定健康")
    
    # 场景4: 完全转移后状态稳定
    print("\n" + "=" * 80)
    print("场景4: 完全转移后 - 状态稳定不再周期性变化")
    print("=" * 80)
    
    interventions_scene4 = [
        (0, 1, 0),  # t=1: 干预
        (0, 1, 0),  # t=2: 干预（触发E2a）
        (0, 0, 0),  # t=3: 无干预
        (0, 0, 0),  # t=4: 无干预
        (0, 0, 0),  # t=5: 无干预
        (0, 0, 0),  # t=6: 无干预
        (0, 1, 0),  # t=7: 干预（不再影响稳定状态）
    ]
    
    result4 = model.simulate(interventions_scene4, return_details=True)
    _print_results(result4, model, title="完全转移后，健康状态稳定不变")


def demo_periodic_visualization():
    """可视化周期性变化"""
    print("\n" + "=" * 80)
    print("周期性变化可视化")
    print("=" * 80)
    
    model = PeriodicStateTransitionModel(T=3, n=2)
    
    # 模拟50步无干预，观察周期性
    interventions = [(0, 0, 0)] * 20
    result = model.simulate(interventions)
    
    # 绘制状态序列
    states = result["states"]
    
    print("\n状态序列（0=健康, 1=不健康）:")
    print("-" * 50)
    
    state_symbols = []
    for i, state in enumerate(states):
        if state == model.HEALTHY:
            symbol = "🟢"  # 健康
            state_symbols.append("0")
        else:
            symbol = "🔴"  # 不健康
            state_symbols.append("1")
        print(f"{symbol}", end=" ")
        
        if (i + 1) % 10 == 0:
            print(f" 步数: {i}")
    
    print("\n")
    print("周期规律: 🔴🔴🔴🟢🟢🟢🔴🔴🔴🟢🟢🟢... (T=3)")


def _print_results(result: dict, model, title: str = ""):
    """打印模拟结果"""
    if title:
        print(f"\n{title}")
    
    print(f"\n干预序列:")
    for t, inter in enumerate(result["interventions"], 1):
        if inter == model.INTERVENTION:
            print(f"   t={t}: {inter} 🔵 干预")
        else:
            print(f"   t={t}: {inter} ⚪ 无干预")
    
    print(f"\n状态演变:")
    print("-" * 70)
    print(f"{'步数':<6} {'可观测状态':<20} {'状态说明':<15} {'事件':<30}")
    print("-" * 70)
    
    for i in range(len(result["states"])):
        state = result["states"][i]
        state_name = "健康" if state == model.HEALTHY else "不健康"
        
        if i == 0:
            event_str = "-"
        else:
            event = result["events"][i-1]
            if event == EventType.COMPLETE_TRANSITION:
                event_str = "E2a (完全转移)"
            elif event == EventType.PERIODIC_TO_UNHEALTHY:
                event_str = "E2b (周期性→不健康)"
            elif event == EventType.PERIODIC_TO_HEALTHY:
                event_str = "E2c (周期性→健康)"
            else:
                event_str = "E2d (状态维持)"
        
        display_state = f"{state} ({state_name})"
        print(f"{i:<6} {display_state:<20} {state_name:<15} {event_str:<30}")
    
    # 显示内部状态（调试用）
    internal = model.get_internal_state()
    if internal["stable_healthy"]:
        print(f"\n💡 已进入稳定健康状态，周期性停止")


if __name__ == "__main__":
    demo_periodic_model()
    demo_periodic_visualization()