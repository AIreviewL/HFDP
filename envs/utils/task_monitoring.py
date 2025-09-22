import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import json
import os

class TaskCompletionVerifier(nn.Module):
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        self.verification_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.task_state_tracker = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.completion_evaluator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                current_features: torch.Tensor,
                task_history: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        direct_verification = self.verification_network(current_features)
        
        if task_history is not None:
            lstm_output, (hidden, cell) = self.task_state_tracker(task_history)
            sequence_verification = self.completion_evaluator(hidden[-1])
        else:
            sequence_verification = torch.zeros_like(direct_verification)
        
        combined_verification = (direct_verification + sequence_verification) / 2
        
        return {
            'verification_score': combined_verification,
            'direct_verification': direct_verification,
            'sequence_verification': sequence_verification,
            'is_completed': combined_verification > 0.8
        }

class HumanFeedbackSystem(nn.Module):
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        self.feedback_encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for feedback signal
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.feedback_quality_evaluator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.feedback_integrator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.feedback_history = []
        
    def forward(self, 
                current_state: torch.Tensor,
                human_feedback: torch.Tensor,
                feedback_metadata: Optional[Dict] = None) -> Dict[str, torch.Tensor]:

        feedback_record = {
            'timestamp': datetime.now().isoformat(),
            'feedback_value': human_feedback.item() if torch.is_tensor(human_feedback) else human_feedback,
            'metadata': feedback_metadata
        }
        self.feedback_history.append(feedback_record)
        
        feedback_enhanced_state = torch.cat([current_state, human_feedback.unsqueeze(-1)], dim=-1)
        encoded_feedback = self.feedback_encoder(feedback_enhanced_state)
        
        feedback_quality = self.feedback_quality_evaluator(encoded_feedback)
        
        integrated_feedback = self.feedback_integrator(encoded_feedback)
        
        return {
            'integrated_feedback': integrated_feedback,
            'feedback_quality': feedback_quality,
            'encoded_feedback': encoded_feedback,
            'feedback_history': self.feedback_history
        }
    
    def save_feedback_history(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_history, f, ensure_ascii=False, indent=2)
    
    def load_feedback_history(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.feedback_history = json.load(f)

class MonitoringSystem(nn.Module):
    
    def __init__(self, 
                 vision_dim: int = 512,
                 language_dim: int = 768,
                 robot_state_dim: int = 64,
                 hidden_dim: int = 256):
        super().__init__()
        
        from .multimodal_encoder import VisionLanguageModel
        
        self.vlm = VisionLanguageModel(vision_dim, language_dim, hidden_dim)
        self.task_verifier = TaskCompletionVerifier(hidden_dim, hidden_dim)
        self.feedback_system = HumanFeedbackSystem(hidden_dim, hidden_dim // 2)
        
        self.monitoring_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # VLM + Verifier + Feedback
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.monitoring_log = []
        
    def forward(self,
                vision_features: torch.Tensor,
                language_features: torch.Tensor,
                robot_state: torch.Tensor,
                task_history: Optional[torch.Tensor] = None,
                human_feedback: Optional[torch.Tensor] = None) -> Dict[str, Any]:

        vlm_output = self.vlm(vision_features, language_features)
        
        verification_output = self.task_verifier(vlm_output, task_history)
        
        if human_feedback is not None:
            feedback_output = self.feedback_system(vlm_output, human_feedback)
        else:
            feedback_output = {
                'integrated_feedback': torch.zeros_like(vlm_output),
                'feedback_quality': torch.zeros(vlm_output.shape[0], 1, device=vlm_output.device),
                'encoded_feedback': torch.zeros_like(vlm_output)
            }
        
        monitoring_features = torch.cat([
            vlm_output,
            verification_output['verification_score'],
            feedback_output['integrated_feedback']
        ], dim=-1)
        
        final_monitoring_score = self.monitoring_fusion(monitoring_features)
        
        monitoring_record = {
            'timestamp': datetime.now().isoformat(),
            'vlm_score': vlm_output.item(),
            'verification_score': verification_output['verification_score'].item(),
            'feedback_quality': feedback_output['feedback_quality'].item(),
            'final_score': final_monitoring_score.item(),
            'is_completed': verification_output['is_completed'].item()
        }
        self.monitoring_log.append(monitoring_record)
        
        return {
            'final_monitoring_score': final_monitoring_score,
            'vlm_output': vlm_output,
            'verification_output': verification_output,
            'feedback_output': feedback_output,
            'monitoring_log': self.monitoring_log,
            'task_completed': final_monitoring_score > 0.9
        }
    
    def save_monitoring_log(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.monitoring_log, f, ensure_ascii=False, indent=2)
    
    def get_task_progress(self) -> Dict[str, float]:
        if not self.monitoring_log:
            return {'completion_rate': 0.0, 'average_score': 0.0}
        
        scores = [record['final_score'] for record in self.monitoring_log]
        completions = [record['is_completed'] for record in self.monitoring_log]
        
        return {
            'completion_rate': sum(completions) / len(completions),
            'average_score': sum(scores) / len(scores),
            'total_steps': len(self.monitoring_log)
        }
