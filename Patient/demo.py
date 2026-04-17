import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delayed_model import DelayedStateTransitionModel
from utils.helpers import format_state_sequence, save_simulation_result


def demo_delayed_model():
    """演示延迟状态转移模型"""
    print("=" * 60)
    print("延迟状态转移模型演示")
    print("=" * 60)
    
    # 创建模型实例
    model = DelayedStateTransitionModel(n=2, m=4, k=2)
    
    # 查看模型信息
    info = model.get_model_info()
    print(f"\n模型信息:")
    print(f"  名称: {info['model_name']}")
    print(f"  参数: {info['model_params']}")
    print(f"  初始状态: {info['initial_state']}")
    print(f"  可用干预: {info['available_interventions']}")
    
    # 定义干预序列
    interventions = [
        (0, 0, 1),  # step 1: 施加干预
        (0, 0, 1),  # step 2: 继续干预（达到n=2，进入部分转移）
        (0, 0, 0),  # step 3: 停止干预
        (0, 0, 0),  # step 4: 仍然没有干预（仍在k=2步内）
        (0, 0, 0),  # step 5: 超过k步，回到不健康
        (0, 0, 1),  # step 6: 重新干预
        (0, 0, 1),  # step 7: 继续干预
        (0, 0, 1),  # step 8: 继续干预
        (0, 0, 1),  # step 9: 继续干预（达到m=4，完全转移）
    ]
    
    # 模拟（返回详细信息）
    result = model.simulate(interventions, return_details=True)
    
    # 格式化输出
    print("\n模拟结果:")
    print(format_state_sequence(
        result["states"],
        result["interventions"],
        result["events"]
    ))
    
    # 保存结果
    save_simulation_result(result, "delayed_model_result.json")
    print("\n结果已保存到 delayed_model_result.json")


def demo_multiple_scenarios():
    """演示多个场景"""
    print("\n" + "=" * 60)
    print("多场景测试")
    print("=" * 60)
    
    model = DelayedStateTransitionModel(n=2, m=4, k=2)
    
    scenarios = [
        {
            "name": "场景1: 无干预",
            "interventions": [(0, 0, 0)] * 5
        },
        {
            "name": "场景2: 部分转移后停止",
            "interventions": [(0, 0, 1)] * 2 + [(0, 0, 0)] * 3
        },
        {
            "name": "场景3: 完全转移",
            "interventions": [(0, 0, 1)] * 4
        },
        {
            "name": "场景4: 部分转移 - 继续干预 - 完全转移",
            "interventions": [(0, 0, 1)] * 6
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        result = model.simulate(scenario["interventions"])
        for t, state in enumerate(result["states"]):
            print(f"  t={t}: {state}")


if __name__ == "__main__":
    demo_delayed_model()
    demo_multiple_scenarios()