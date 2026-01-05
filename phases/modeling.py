"""Modeling Phase - Model training and evaluation"""

from typing import TYPE_CHECKING
from core.memory import ModelRecord
import json

if TYPE_CHECKING:
    from core.llm import LLMManager
    from core.executor import CodeExecutor
    from core.memory import AgentMemory


BASELINE_PROMPT = """You are a Kaggle Grandmaster. Generate Python code for a baseline model.

## Competition Info:
- Task: {task_type}
- Metric: {metric}
- Target column: {target_column}
- Data directory: {data_dir}

## EDA Results:
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

## Requirements:
1. Load train.csv (or train_fe.csv if exists)
2. Preprocessing (missing value imputation, label encoding)
3. Train LightGBM with 5-fold CV
4. Save predictions for test.csv as submission.csv

## Output Format:
```
print(json.dumps({{
    "cv_score": cv_score,
    "fold_scores": [score1, score2, score3, score4, score5],
    "model_type": "lightgbm"
}}))
```

Generate only the code without explanations.
"""


IMPROVE_PROMPT = """You are a Kaggle Grandmaster. Improve the model based on the following info.

## Competition Info:
- Task: {task_type}
- Metric: {metric}
- Current best score: {best_score}

## New Features:
{new_features}

## Previous Experiments:
{experiment_history}

## Improvement Ideas:
- Hyperparameter tuning (learning_rate, num_leaves, etc.)
- Try XGBoost or CatBoost
- Feature selection based on importance
- Different CV strategy (StratifiedKFold, GroupKFold)

## Requirements:
1. Load train_fe.csv (or train.csv)
2. Apply improvements
3. Train with 5-fold CV
4. Save predictions as submission.csv

## Output Format:
```
print(json.dumps({{
    "cv_score": cv_score,
    "model_type": "model_name",
    "improvement": "what was improved"
}}))
```

Generate only the code without explanations.
"""


class ModelingPhase:
    """Modeling Phase"""
    
    def __init__(self, llm: "LLMManager", executor: "CodeExecutor", memory: "AgentMemory"):
        self.llm = llm
        self.executor = executor
        self.memory = memory
    
    def create_baseline(self, eda_insights: dict) -> dict:
        context = self.memory.competition_context
        
        prompt = BASELINE_PROMPT.format(
            task_type=context.get("task_type", "classification"),
            metric=context.get("metric", "auc"),
            target_column=context.get("target_column", "target"),
            data_dir=context.get("data_dir", "."),
            numeric_columns=eda_insights.get("numeric_columns", []),
            categorical_columns=eda_insights.get("categorical_columns", [])
        )
        
        code = self.llm.generate_code(prompt)
        result = self.executor.execute_with_retry(code, self.llm)
        
        if result.success:
            model_result = self._parse_output(result.stdout)
            cv_score = model_result.get("cv_score", 0)
            
            self.memory.add_model(ModelRecord(
                name="baseline_lgb",
                model_type="lightgbm",
                cv_score=cv_score
            ))
            
            self.memory.add_experiment(
                phase="modeling",
                description="Baseline LightGBM",
                code=code,
                cv_score=cv_score,
                success=True,
                model_type="lightgbm"
            )
            return model_result
        else:
            self.memory.add_experiment(
                phase="modeling",
                description="Baseline (failed)",
                code=code,
                success=False
            )
            return {"cv_score": 0, "model_type": "failed"}
    
    def improve(self, fe_result: dict) -> dict:
        context = self.memory.competition_context
        
        prompt = IMPROVE_PROMPT.format(
            task_type=context.get("task_type", "classification"),
            metric=context.get("metric", "auc"),
            best_score=self.memory.best_score,
            new_features=fe_result.get("new_features", []),
            experiment_history=self.memory.get_history_summary()
        )
        
        code = self.llm.generate_code(prompt)
        result = self.executor.execute_with_retry(code, self.llm)
        
        if result.success:
            model_result = self._parse_output(result.stdout)
            cv_score = model_result.get("cv_score", 0)
            
            if cv_score > self.memory.best_score:
                self.memory.add_model(ModelRecord(
                    name=f"improved_{model_result.get('model_type', 'unknown')}",
                    model_type=model_result.get("model_type", "unknown"),
                    cv_score=cv_score
                ))
            
            self.memory.add_experiment(
                phase="modeling",
                description=model_result.get("improvement", "Improvement"),
                code=code,
                cv_score=cv_score,
                success=True,
                model_type=model_result.get("model_type")
            )
            return model_result
        else:
            self.memory.add_experiment(
                phase="modeling",
                description="Improvement (failed)",
                code=code,
                success=False
            )
            return {"cv_score": 0}
    
    def _parse_output(self, stdout: str) -> dict:
        lines = stdout.strip().split("\n")
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return {"cv_score": 0, "model_type": "unknown"}
