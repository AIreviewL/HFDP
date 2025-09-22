import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json
import os
from datetime import datetime

from .utils.multimodal_encoder import MultimodalEncoder, VisionLanguageModel
from .utils.control_strategies import ControlStrategyManager
from .utils.task_monitoring import MonitoringSystem
from .robot.enhanced_ik_control import EnhancedIKImpedanceController
from code_gen.enhanced_llm_agent import EnhancedLLMAgent

class EnhancedRobotController(nn.Module):

    
    def __init__(self, 
                 language_dim: int = 768,
                 vision_dim: int = 512,
                 robot_state_dim: int = 64,
                 tactile_dim: int = 32,
                 joint_dim: int = 12,
                 cartesian_dim: int = 6,
                 hidden_dim: int = 512):
        super().__init__()
        
        self.multimodal_encoder = MultimodalEncoder(
            language_dim=language_dim,
            vision_dim=vision_dim,
            robot_state_dim=robot_state_dim,
            tactile_dim=tactile_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        self.control_strategy_manager = ControlStrategyManager(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=cartesian_dim
        )
        
        self.monitoring_system = MonitoringSystem(
            vision_dim=vision_dim,
            language_dim=language_dim,
            robot_state_dim=robot_state_dim,
            hidden_dim=hidden_dim // 2
        )
        
        self.ik_impedance_controller = EnhancedIKImpedanceController(
            joint_dim=joint_dim,
            cartesian_dim=cartesian_dim,
            state_dim=hidden_dim,
            action_dim=joint_dim,
            hidden_dim=hidden_dim
        )
        
        self.llm_agent = EnhancedLLMAgent(
            language_dim=language_dim,
            hidden_dim=hidden_dim // 2,
            max_subtasks=10
        )
        
        self.task_state_tracker = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.system_state_manager = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.execution_history = []
        self.task_history = []
        
    def forward(self,
                language_instruction: str,
                vision_features: torch.Tensor,
                robot_state: torch.Tensor,
                tactile_features: torch.Tensor,
                current_joints: torch.Tensor,
                current_pose: torch.Tensor,
                target_force: torch.Tensor,
                current_force: torch.Tensor,
                human_feedback: Optional[torch.Tensor] = None,
                task_context: Optional[Dict] = None) -> Dict[str, Any]:

        
        if task_context is None or 'subtasks' not in task_context:
            task_decomposition = self.llm_agent.decompose_task(language_instruction, task_context)
            task_context = {'subtasks': task_decomposition}
        
        multimodal_output = self.multimodal_encoder(
            language_features=vision_features,  # 简化处理，实际需要语言编码
            vision_features=vision_features,
            robot_state=robot_state,
            tactile_features=tactile_features
        )
        
        control_strategy_output = self.control_strategy_manager(
            features=multimodal_output['fused_features'],
            tactile_data=tactile_features,
            robot_state=robot_state,
            human_feedback=human_feedback
        )
        
        monitoring_output = self.monitoring_system(
            vision_features=vision_features,
            language_features=vision_features,  # 简化处理
            robot_state=robot_state,
            task_history=self._get_task_history(),
            human_feedback=human_feedback
        )
        
        ik_impedance_output = self.ik_impedance_controller(
            target_pose=control_strategy_output['final_control_output'],
            current_joints=current_joints,
            current_pose=current_pose,
            target_force=target_force,
            current_force=current_force,
            state=multimodal_output['fused_features'],
            human_feedback=human_feedback
        )
        
        system_state = self.system_state_manager(
            torch.cat([
                multimodal_output['fused_features'],
                control_strategy_output['final_control_output'],
                monitoring_output['final_monitoring_score']
            ], dim=-1)
        )
        
        self._update_task_history(system_state)
        
        execution_record = {
            'timestamp': datetime.now().isoformat(),
            'language_instruction': language_instruction,
            'system_state': system_state.detach().cpu().numpy().tolist(),
            'control_output': ik_impedance_output['final_joint_targets'].detach().cpu().numpy().tolist(),
            'monitoring_score': monitoring_output['final_monitoring_score'].item(),
            'task_completed': monitoring_output['task_completed'],
            'human_feedback': human_feedback.item() if human_feedback is not None else None
        }
        self.execution_history.append(execution_record)
        
        return {
            'joint_targets': ik_impedance_output['final_joint_targets'],
            'task_decomposition': task_context['subtasks'],
            'multimodal_features': multimodal_output,
            'control_strategy': control_strategy_output,
            'monitoring_result': monitoring_output,
            'ik_impedance_result': ik_impedance_output,
            'system_state': system_state,
            'task_completed': monitoring_output['task_completed'],
            'execution_history': self.execution_history
        }
    
    def _get_task_history(self) -> Optional[torch.Tensor]:
        if len(self.task_history) == 0:
            return None
        
        recent_history = self.task_history[-10:]  
        history_tensor = torch.stack(recent_history)
        return history_tensor.unsqueeze(0)  
    
    def _update_task_history(self, current_state: torch.Tensor):
        self.task_history.append(current_state)
        
        if len(self.task_history) > 100:
            self.task_history = self.task_history[-100:]
    
    def monitor_subtask_execution(self, 
                                subtask: Dict,
                                execution_context: Dict,
                                visual_observation: Optional[str] = None) -> Dict[str, Any]:
        return self.llm_agent.monitor_subtask_execution(
            subtask, execution_context, visual_observation
        )
    
    def verify_task_completion(self, 
                             task_instruction: str,
                             execution_summary: Dict) -> Dict[str, Any]:
        return self.llm_agent.generate_task_completion_verification(
            task_instruction, execution_summary
        )
    
    def save_execution_data(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        
        execution_file = os.path.join(save_dir, 'execution_history.json')
        with open(execution_file, 'w', encoding='utf-8') as f:
            json.dump(self.execution_history, f, ensure_ascii=False, indent=2)
        
        task_file = os.path.join(save_dir, 'task_history.json')
        task_history_data = [state.detach().cpu().numpy().tolist() for state in self.task_history]
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task_history_data, f, ensure_ascii=False, indent=2)
        
        llm_dir = os.path.join(save_dir, 'llm_agent')
        os.makedirs(llm_dir, exist_ok=True)
        self.llm_agent.save_task_history(os.path.join(llm_dir, 'task_history.json'))
        self.llm_agent.save_execution_log(os.path.join(llm_dir, 'execution_log.json'))
        
        monitoring_file = os.path.join(save_dir, 'monitoring_log.json')
        self.monitoring_system.save_monitoring_log(monitoring_file)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for record in self.execution_history 
                                  if record.get('task_completed', False))
        
        monitoring_stats = self.monitoring_system.get_task_progress()
        
        llm_stats = self.llm_agent.get_task_statistics()
        
        return {
            'execution_stats': {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'success_rate': successful_executions / total_executions if total_executions > 0 else 0.0
            },
            'monitoring_stats': monitoring_stats,
            'llm_stats': llm_stats,
            'task_history_length': len(self.task_history)
        }
    
    def reset_system(self):
        self.execution_history = []
        self.task_history = []
        self.llm_agent.task_history = []
        self.llm_agent.subtask_execution_log = []
        self.monitoring_system.monitoring_log = []

class RoboTwinEnhancedController:
    
    def __init__(self, config: Dict[str, Any]):

        self.config = config
        
        self.enhanced_controller = EnhancedRobotController(
            language_dim=config.get('language_dim', 768),
            vision_dim=config.get('vision_dim', 512),
            robot_state_dim=config.get('robot_state_dim', 64),
            tactile_dim=config.get('tactile_dim', 32),
            joint_dim=config.get('joint_dim', 12),
            cartesian_dim=config.get('cartesian_dim', 6),
            hidden_dim=config.get('hidden_dim', 512)
        )
        
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.enhanced_controller.to(self.device)
        
        self.training_mode = config.get('training_mode', False)
        if self.training_mode:
            self.enhanced_controller.train()
        else:
            self.enhanced_controller.eval()
    
    def execute_task(self, 
                    task_instruction: str,
                    observation: Dict[str, torch.Tensor],
                    human_feedback: Optional[torch.Tensor] = None) -> Dict[str, Any]:

        with torch.no_grad() if not self.training_mode else torch.enable_grad():
            vision_features = observation.get('vision', torch.zeros(1, 512)).to(self.device)
            robot_state = observation.get('robot_state', torch.zeros(1, 64)).to(self.device)
            tactile_features = observation.get('tactile', torch.zeros(1, 32)).to(self.device)
            current_joints = observation.get('joints', torch.zeros(1, 12)).to(self.device)
            current_pose = observation.get('pose', torch.zeros(1, 6)).to(self.device)
            target_force = observation.get('target_force', torch.zeros(1, 6)).to(self.device)
            current_force = observation.get('current_force', torch.zeros(1, 6)).to(self.device)
            
            if human_feedback is not None:
                human_feedback = human_feedback.to(self.device)
            
            result = self.enhanced_controller(
                language_instruction=task_instruction,
                vision_features=vision_features,
                robot_state=robot_state,
                tactile_features=tactile_features,
                current_joints=current_joints,
                current_pose=current_pose,
                target_force=target_force,
                current_force=current_force,
                human_feedback=human_feedback
            )
            
            return result
    
    def save_system_state(self, save_dir: str):
        self.enhanced_controller.save_execution_data(save_dir)
    
    def get_statistics(self) -> Dict[str, Any]:
        return self.enhanced_controller.get_system_statistics()
    
    def reset(self):
        self.enhanced_controller.reset_system()
