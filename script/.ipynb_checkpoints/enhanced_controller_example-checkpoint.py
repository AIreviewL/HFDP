#!/usr/bin/env python3
"""
RoboTwin增强控制器使用示例
展示如何使用新的架构图改进的控制器
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Any

# 导入增强控制器
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.enhanced_robot_controller import RoboTwinEnhancedController

def create_sample_observation() -> Dict[str, torch.Tensor]:
    """创建示例观察数据"""
    return {
        'vision': torch.randn(1, 512),  # 视觉特征
        'robot_state': torch.randn(1, 64),  # 机器人状态
        'tactile': torch.randn(1, 32),  # 触觉特征
        'joints': torch.randn(1, 12),  # 关节角度
        'pose': torch.randn(1, 6),  # 位姿
        'target_force': torch.randn(1, 6),  # 目标力
        'current_force': torch.randn(1, 6)  # 当前力
    }

def create_controller_config() -> Dict[str, Any]:
    """创建控制器配置"""
    return {
        'language_dim': 768,
        'vision_dim': 512,
        'robot_state_dim': 64,
        'tactile_dim': 32,
        'joint_dim': 12,
        'cartesian_dim': 6,
        'hidden_dim': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'training_mode': False
    }

def demonstrate_task_execution():
    """演示任务执行流程"""
    print("=== RoboTwin增强控制器演示 ===\n")
    
    # 1. 创建控制器
    print("1. 初始化增强控制器...")
    config = create_controller_config()
    controller = RoboTwinEnhancedController(config)
    print(f"   控制器已创建，设备: {config['device']}")
    
    # 2. 准备任务数据
    print("\n2. 准备任务数据...")
    task_instruction = "将红色积木放在蓝色积木上面"
    observation = create_sample_observation()
    human_feedback = torch.tensor([0.8])  # 正面反馈
    
    print(f"   任务指令: {task_instruction}")
    print(f"   观察数据维度: {observation['vision'].shape}")
    print(f"   人类反馈: {human_feedback.item()}")
    
    # 3. 执行任务
    print("\n3. 执行任务...")
    try:
        result = controller.execute_task(
            task_instruction=task_instruction,
            observation=observation,
            human_feedback=human_feedback
        )
        
        print("   任务执行成功!")
        print(f"   关节目标: {result['joint_targets'].shape}")
        print(f"   任务完成: {result['task_completed']}")
        print(f"   监控分数: {result['monitoring_result']['final_monitoring_score'].item():.3f}")
        
        # 显示子任务分解
        subtasks = result['task_decomposition']
        if isinstance(subtasks, dict) and 'subtasks' in subtasks:
            print(f"   子任务数量: {len(subtasks['subtasks'])}")
            for i, subtask in enumerate(subtasks['subtasks'][:3]):  # 显示前3个
                print(f"     子任务{i+1}: {subtask.get('description', 'N/A')}")
        
    except Exception as e:
        print(f"   任务执行失败: {e}")
        return
    
    # 4. 获取系统统计
    print("\n4. 获取系统统计...")
    stats = controller.get_statistics()
    print(f"   执行统计: {stats['execution_stats']}")
    print(f"   监控统计: {stats['monitoring_stats']}")
    print(f"   LLM统计: {stats['llm_stats']}")
    
    # 5. 保存系统状态
    print("\n5. 保存系统状态...")
    save_dir = f"enhanced_controller_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    controller.save_system_state(save_dir)
    print(f"   数据已保存到: {save_dir}")
    
    print("\n=== 演示完成 ===")

def demonstrate_subtask_monitoring():
    """演示子任务监控功能"""
    print("\n=== 子任务监控演示 ===\n")
    
    # 创建控制器
    config = create_controller_config()
    controller = RoboTwinEnhancedController(config)
    
    # 示例子任务
    subtask = {
        'id': 'subtask_1',
        'description': '抓取红色积木',
        'goal': '成功抓取红色积木',
        'expected_result': '红色积木被抓取'
    }
    
    # 执行上下文
    execution_context = {
        'robot_state': '准备抓取',
        'environment_state': '积木在桌面上',
        'progress': '50%'
    }
    
    # 视觉观察
    visual_observation = "机器人手臂正在接近红色积木，距离约5cm"
    
    print("监控子任务执行...")
    monitoring_result = controller.enhanced_controller.monitor_subtask_execution(
        subtask=subtask,
        execution_context=execution_context,
        visual_observation=visual_observation
    )
    
    print(f"执行状态: {monitoring_result.get('execution_status', 'unknown')}")
    print(f"完成度: {monitoring_result.get('completion_percentage', 0):.1%}")
    print(f"问题: {monitoring_result.get('issues', [])}")
    print(f"建议: {monitoring_result.get('recommendations', [])}")

def demonstrate_task_verification():
    """演示任务完成验证功能"""
    print("\n=== 任务完成验证演示 ===\n")
    
    # 创建控制器
    config = create_controller_config()
    controller = RoboTwinEnhancedController(config)
    
    # 原始任务指令
    task_instruction = "将红色积木放在蓝色积木上面"
    
    # 执行总结
    execution_summary = {
        'subtasks_completed': ['抓取红色积木', '移动到蓝色积木位置', '放置积木'],
        'final_state': '红色积木成功放置在蓝色积木上',
        'execution_time': '15秒',
        'success_rate': 0.95
    }
    
    print("验证任务完成...")
    verification_result = controller.enhanced_controller.verify_task_completion(
        task_instruction=task_instruction,
        execution_summary=execution_summary
    )
    
    print(f"任务完成: {verification_result.get('task_completed', False)}")
    print(f"完成分数: {verification_result.get('completion_score', 0):.3f}")
    print(f"结果质量: {verification_result.get('final_result_quality', 'unknown')}")
    print(f"验证说明: {verification_result.get('verification_notes', 'N/A')}")

def demonstrate_human_feedback_integration():
    """演示人类反馈集成功能"""
    print("\n=== 人类反馈集成演示 ===\n")
    
    # 创建控制器
    config = create_controller_config()
    controller = RoboTwinEnhancedController(config)
    
    # 不同的人类反馈
    feedback_scenarios = [
        (torch.tensor([1.0]), "强烈正面反馈"),
        (torch.tensor([0.5]), "中等正面反馈"),
        (torch.tensor([0.0]), "中性反馈"),
        (torch.tensor([-0.5]), "中等负面反馈"),
        (torch.tensor([-1.0]), "强烈负面反馈")
    ]
    
    observation = create_sample_observation()
    task_instruction = "调整机器人动作"
    
    for feedback, description in feedback_scenarios:
        print(f"测试 {description}...")
        
        result = controller.execute_task(
            task_instruction=task_instruction,
            observation=observation,
            human_feedback=feedback
        )
        
        # 检查人类反馈是否被集成
        feedback_integrated = result['ik_impedance_result']['human_feedback_integrated']
        print(f"  人类反馈集成: {feedback_integrated}")
        
        # 显示控制输出差异
        control_output = result['joint_targets']
        print(f"  控制输出范围: [{control_output.min().item():.3f}, {control_output.max().item():.3f}]")

def main():
    """主函数"""
    print("RoboTwin增强控制器功能演示")
    print("=" * 50)
    
    try:
        # 演示基本任务执行
        demonstrate_task_execution()
        
        # 演示子任务监控
        demonstrate_subtask_monitoring()
        
        # 演示任务验证
        demonstrate_task_verification()
        
        # 演示人类反馈集成
        demonstrate_human_feedback_integration()
        
        print("\n所有演示完成!")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
