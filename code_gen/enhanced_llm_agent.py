import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from datetime import datetime
from .gpt_agent import generate

class EnhancedLLMAgent(nn.Module):
    
    def __init__(self, 
                 language_dim: int = 768,
                 hidden_dim: int = 512,
                 max_subtasks: int = 10):
        super().__init__()
        
        self.task_decomposition_network = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_subtasks * 256)  
        )
        
        self.subtask_priority_evaluator = nn.Sequential(
            nn.Linear(256, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dependency_model = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.monitoring_prompt_generator = nn.Sequential(
            nn.Linear(language_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, language_dim)
        )
        
        self.task_history = []
        self.subtask_execution_log = []
        
    def decompose_task(self, task_instruction: str, context: Optional[Dict] = None) -> Dict[str, Any]:
            
        decomposition_prompt = f"""
        Please decompose the following robot task into specific subtasks:
        
        Task: {task_instruction}
        
        Requirements:
        1. Each subtask should be concrete and executable
        2. Consider the dependencies between subtasks
        3. Each subtask must include goal, constraints, and expected result
        4. The number of subtasks should be between 3 and 8
        
        Please return in JSON format:
        {{
            "subtasks": [
                {{
                    "id": "subtask_1",
                    "description": "Subtask description",
                    "goal": "Specific goal",
                    "constraints": ["Constraint 1", "Constraint 2"],
                    "expected_result": "Expected result",
                    "dependencies": ["Dependent subtask IDs"],
                    "priority": 1
                }}
            ],
            "execution_order": ["subtask_1", "subtask_2", ...],
            "estimated_duration": "Estimated total duration"
        }}
        """
        
        try:
            response = generate([{"role": "user", "content": decomposition_prompt}], temperature=0.1)
            decomposition_result = json.loads(response)
            
            decomposition_record = {
                'timestamp': datetime.now().isoformat(),
                'task_instruction': task_instruction,
                'context': context,
                'decomposition_result': decomposition_result
            }
            self.task_history.append(decomposition_record)
            
            return decomposition_result
            
        except Exception as e:
            print(f"Task decomposition failed: {e}")
            # Return a default decomposition
            return self._default_task_decomposition(task_instruction)

    
    def _default_task_decomposition(self, task_instruction: str) -> Dict[str, Any]:
        return {
            "subtasks": [
                {
                    "id": "subtask_1",
                    "description": f"Execute task: {task_instruction}",
                    "goal": task_instruction,
                    "constraints": [],
                    "expected_result": "Task completed",
                    "dependencies": [],
                    "priority": 1
                }
            ],
            "execution_order": ["subtask_1"],
            "estimated_duration": "Unknown"
        }

    
    def generate_monitoring_prompt(self, 
                                current_subtask: Dict,
                                execution_context: Dict) -> str:

        monitoring_prompt = f"""
        Please monitor the execution of the following subtask:
        
        Subtask: {current_subtask['description']}
        Goal: {current_subtask['goal']}
        Expected Result: {current_subtask['expected_result']}
        
        Current Execution Context:
        - Robot State: {execution_context.get('robot_state', 'Unknown')}
        - Environment State: {execution_context.get('environment_state', 'Unknown')}
        - Progress: {execution_context.get('progress', 'Unknown')}
        
        Please evaluate:
        1. Is the subtask being executed as expected?
        2. Are there any anomalies or deviations?
        3. Is it necessary to adjust the execution strategy?
        4. Has the subtask been completed?
        
        Please return the evaluation results in JSON format:
        {{
            "execution_status": "normal/abnormal/completed",
            "completion_percentage": 0.85,
            "issues": ["Issue 1", "Issue 2"],
            "recommendations": ["Recommendation 1", "Recommendation 2"],
            "next_actions": ["Next action 1", "Next action 2"]
        }}
        """
        
        return monitoring_prompt

    
    def monitor_subtask_execution(self, 
                                subtask: Dict,
                                execution_context: Dict,
                                visual_observation: Optional[str] = None) -> Dict[str, Any]:

        monitoring_prompt = self.generate_monitoring_prompt(subtask, execution_context)
        
        if visual_observation:
            monitoring_prompt += f"\n\nVisual observation：{visual_observation}"
        
        try:
            response = generate([{"role": "user", "content": monitoring_prompt}], temperature=0.1)
            monitoring_result = json.loads(response)
            

            monitoring_record = {
                'timestamp': datetime.now().isoformat(),
                'subtask_id': subtask['id'],
                'execution_context': execution_context,
                'monitoring_result': monitoring_result
            }
            self.subtask_execution_log.append(monitoring_record)
            
            return monitoring_result
            
        except Exception as e:
            print(f"Monitoring Failure: {e}")
            return {
                "execution_status": "unknown",
                "completion_percentage": 0.0,
                "issues": ["Monitoring Failure"],
                "recommendations": ["Checking system status"],
                "next_actions": ["Continue execution"]
            }
    
    def generate_task_completion_verification(self, 
                                            task_instruction: str,
                                            execution_summary: Dict) -> Dict[str, Any]:

        verification_prompt = f"""
        Please verify that the following tasks are completed:
        
        Original Task：{task_instruction}
        
        Executive Summary:
        {json.dumps(execution_summary, ensure_ascii=False, indent=2)}
        
        Please evaluate:
        1. Was the task performed exactly as instructed?
        2. Were all subtasks completed?
        3. Did the final result meet expectations?
        4. Were there any unfinished tasks?
        
        Please return the verification result in JSON format:
        {{
            "task_completed": true/false,
            "completion_score": 0.95,
            "completed_subtasks": ["subtask_1", "subtask_2"],
            "incomplete_subtasks": [],
            "final_result_quality": "excellent/good/fair/poor",
            "verification_notes": "Verification Instructions"
        }}
        """
        
        try:
            response = generate([{"role": "user", "content": verification_prompt}], temperature=0.1)
            verification_result = json.loads(response)
            return verification_result
            
        except Exception as e:
            print(f"Verification Failed: {e}")
            return {
                "task_completed": False,
                "completion_score": 0.0,
                "completed_subtasks": [],
                "incomplete_subtasks": [],
                "final_result_quality": "unknown",
                "verification_notes": "Verification Failed"
            }
    
    def save_task_history(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.task_history, f, ensure_ascii=False, indent=2)
    
    def save_execution_log(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.subtask_execution_log, f, ensure_ascii=False, indent=2)
    
    def get_task_statistics(self) -> Dict[str, Any]:
        if not self.task_history:
            return {'total_tasks': 0, 'success_rate': 0.0}
        
        total_tasks = len(self.task_history)
        successful_tasks = sum(1 for record in self.task_history 
                             if record.get('decomposition_result', {}).get('status') == 'success')
        
        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            'average_subtasks_per_task': sum(len(record.get('decomposition_result', {}).get('subtasks', [])) 
                                           for record in self.task_history) / total_tasks if total_tasks > 0 else 0.0
        }
