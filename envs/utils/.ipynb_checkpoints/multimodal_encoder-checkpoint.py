import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class MultimodalEncoder(nn.Module):

    def __init__(self, 
                 language_dim: int = 768,
                 vision_dim: int = 512,
                 robot_state_dim: int = 64,
                 tactile_dim: int = 32,
                 hidden_dim: int = 1024,
                 output_dim: int = 512):
        super().__init__()
        

        self.language_encoder = nn.Linear(language_dim, hidden_dim)
        
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.robot_state_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.tactile_encoder = nn.Sequential(
            nn.Linear(tactile_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.subtask_head = nn.Linear(output_dim, 256)  
        self.trajectory_head = nn.Linear(output_dim, 256)  
        self.reward_head = nn.Linear(output_dim, 128) 
        
    def forward(self, 
                language_features: torch.Tensor,
                vision_features: torch.Tensor,
                robot_state: torch.Tensor,
                tactile_features: torch.Tensor) -> Dict[str, torch.Tensor]:

        lang_encoded = self.language_encoder(language_features)
        vision_encoded = self.vision_encoder(vision_features)
        robot_encoded = self.robot_state_encoder(robot_state)
        tactile_encoded = self.tactile_encoder(tactile_features)
        
        combined = torch.cat([lang_encoded, vision_encoded, robot_encoded, tactile_encoded], dim=-1)
        fused_features = self.fusion_layer(combined)
        
        subtasks = self.subtask_head(fused_features)
        candidate_trajectories = self.trajectory_head(fused_features)
        reward = self.reward_head(fused_features)
        
        return {
            'subtasks': subtasks,
            'candidate_trajectories': candidate_trajectories,
            'reward': reward,
            'fused_features': fused_features
        }

class VisionLanguageModel(nn.Module):

    def __init__(self, vision_dim: int = 512, language_dim: int = 768, output_dim: int = 256):
        super().__init__()
        
        self.vision_encoder = nn.Linear(vision_dim, output_dim)
        self.language_encoder = nn.Linear(language_dim, output_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.task_completion_head = nn.Linear(output_dim, 1)  
        
    def forward(self, vision_features: torch.Tensor, language_features: torch.Tensor) -> torch.Tensor:

        vision_encoded = self.vision_encoder(vision_features)
        language_encoded = self.language_encoder(language_features)
        
        combined = torch.cat([vision_encoded, language_encoded], dim=-1)
        fused = self.fusion(combined)
        
        completion_prob = torch.sigmoid(self.task_completion_head(fused))
        return completion_prob
