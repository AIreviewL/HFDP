import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import transforms3d as t3d
from scipy.spatial.transform import Rotation as R

class EnhancedIKController(nn.Module):

    
    def __init__(self, 
                 joint_dim: int = 12,  # 双臂机器人关节数
                 cartesian_dim: int = 6,  # 笛卡尔空间维度 (位置+姿态)
                 hidden_dim: int = 256):
        super().__init__()
        
        self.ik_network = nn.Sequential(
            nn.Linear(cartesian_dim + joint_dim, hidden_dim),  # 目标位姿 + 当前关节角度
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, joint_dim)
        )
        
        self.jacobian_estimator = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, cartesian_dim * joint_dim)
        )
        
        self.joint_limit_checker = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, joint_dim),
            nn.Sigmoid()  # 输出关节限制违反概率
        )
        
        self.singularity_detector = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # 输出奇异点概率
        )
        
    def forward(self, 
                target_pose: torch.Tensor,
                current_joints: torch.Tensor,
                joint_limits: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:


        ik_input = torch.cat([target_pose, current_joints], dim=-1)
        

        joint_delta = self.ik_network(ik_input)
        target_joints = current_joints + joint_delta
        
        jacobian_flat = self.jacobian_estimator(current_joints)
        jacobian = jacobian_flat.view(-1, 6, joint_delta.shape[-1])
        
        if joint_limits is not None:
            limit_violations = self.joint_limit_checker(target_joints)
            target_joints = torch.clamp(target_joints, 
                                      joint_limits[:, 0], 
                                      joint_limits[:, 1])
        else:
            limit_violations = torch.zeros_like(target_joints)
        
        singularity_prob = self.singularity_detector(current_joints)
        
        return {
            'target_joints': target_joints,
            'joint_delta': joint_delta,
            'jacobian': jacobian,
            'limit_violations': limit_violations,
            'singularity_probability': singularity_prob
        }

class ImpedanceController(nn.Module):

    def __init__(self, 
                 cartesian_dim: int = 6,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.impedance_learner = nn.Sequential(
            nn.Linear(cartesian_dim * 2, hidden_dim),  # 目标位姿 + 当前位姿
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, cartesian_dim * 3)  # Kp, Kd, Ki (每个维度3个参数)
        )
        
        self.force_processor = nn.Sequential(
            nn.Linear(cartesian_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, cartesian_dim)
        )
        
        self.contact_force_estimator = nn.Sequential(
            nn.Linear(cartesian_dim * 2, hidden_dim),  # 期望力 + 实际力
            nn.ReLU(),
            nn.Linear(hidden_dim, cartesian_dim)
        )
        
    def forward(self,
                target_pose: torch.Tensor,
                current_pose: torch.Tensor,
                target_force: torch.Tensor,
                current_force: torch.Tensor,
                dt: float = 0.01) -> Dict[str, torch.Tensor]:

        pose_error = target_pose - current_pose
        
        impedance_input = torch.cat([target_pose, current_pose], dim=-1)
        impedance_params = self.impedance_learner(impedance_input)
        
        batch_size = target_pose.shape[0]
        Kp = impedance_params[:, :6].view(batch_size, 6)
        Kd = impedance_params[:, 6:12].view(batch_size, 6)
        Ki = impedance_params[:, 12:].view(batch_size, 6)
        
        force_feedback = self.force_processor(current_force)
        
        force_input = torch.cat([target_force, current_force], dim=-1)
        contact_force = self.contact_force_estimator(force_input)
        
        impedance_output = Kp * pose_error + Kd * force_feedback + Ki * contact_force
        
        return {
            'impedance_output': impedance_output,
            'pose_error': pose_error,
            'force_error': target_force - current_force,
            'contact_force': contact_force,
            'impedance_params': {
                'Kp': Kp,
                'Kd': Kd,
                'Ki': Ki
            }
        }

class RewardModel(nn.Module):

    def __init__(self, 
                 state_dim: int = 128,
                 action_dim: int = 64,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.completion_reward_estimator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor,
                next_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        sa_input = torch.cat([state, action], dim=-1)
        
        q_value = self.q_network(sa_input)
        
        reward = self.reward_predictor(sa_input)
        
        completion_reward = self.completion_reward_estimator(state)
        
        if next_state is not None:
            next_sa_input = torch.cat([next_state, torch.zeros_like(action)], dim=-1)
            next_q_value = self.q_network(next_sa_input)
            td_error = reward + 0.99 * next_q_value - q_value
        else:
            td_error = torch.zeros_like(q_value)
        
        return {
            'q_value': q_value,
            'reward': reward,
            'completion_reward': completion_reward,
            'td_error': td_error,
            'total_reward': reward + completion_reward
        }

class EnhancedIKImpedanceController(nn.Module):
    
    def __init__(self, 
                 joint_dim: int = 12,
                 cartesian_dim: int = 6,
                 state_dim: int = 128,
                 action_dim: int = 64,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.ik_controller = EnhancedIKController(joint_dim, cartesian_dim, hidden_dim)
        self.impedance_controller = ImpedanceController(cartesian_dim, hidden_dim)
        self.reward_model = RewardModel(state_dim, action_dim, hidden_dim)
        
        self.control_fusion = nn.Sequential(
            nn.Linear(joint_dim + cartesian_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, joint_dim)
        )
        
        self.human_feedback_integrator = nn.Sequential(
            nn.Linear(joint_dim + 1, hidden_dim // 2),  # +1 for human feedback
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, joint_dim)
        )
        
    def forward(self,
                target_pose: torch.Tensor,
                current_joints: torch.Tensor,
                current_pose: torch.Tensor,
                target_force: torch.Tensor,
                current_force: torch.Tensor,
                state: torch.Tensor,
                human_feedback: Optional[torch.Tensor] = None,
                joint_limits: Optional[torch.Tensor] = None) -> Dict[str, Any]:

        ik_output = self.ik_controller(target_pose, current_joints, joint_limits)
        
        impedance_output = self.impedance_controller(
            target_pose, current_pose, target_force, current_force
        )
        
        control_input = torch.cat([ik_output['target_joints'], impedance_output['impedance_output']], dim=-1)
        fused_control = self.control_fusion(control_input)
        
        if human_feedback is not None:
            feedback_input = torch.cat([fused_control, human_feedback.unsqueeze(-1)], dim=-1)
            final_control = self.human_feedback_integrator(feedback_input)
        else:
            final_control = fused_control
        
        reward_output = self.reward_model(state, final_control)
        
        return {
            'final_joint_targets': final_control,
            'ik_output': ik_output,
            'impedance_output': impedance_output,
            'reward_output': reward_output,
            'fused_control': fused_control,
            'human_feedback_integrated': human_feedback is not None
        }
