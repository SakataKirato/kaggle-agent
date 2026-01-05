"""Ensemble Phase - Ensemble learning"""

from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from core.llm import LLMManager
    from core.executor import CodeExecutor
    from core.memory import AgentMemory, ModelRecord


ENSEMBLE_PROMPT = """You are a Kaggle Grandmaster. Generate ensemble code.

## Competition Info:
- Task: {task_type}
- Metric: {metric}
- Target column: {target_column}
- Data directory: {data_dir}

## Available Models:
{available_models}

## Ensemble Strategy:
{ensemble_strategy}

## Requirements:
1. Train multiple models (LightGBM, XGBoost, CatBoost)
2. Apply ensemble ({ensemble_type})
3. Calculate CV score
4. Save predictions as submission.csv

## Output Format:
```
print(json.dumps({{
    "cv_score": cv_score,
    "ensemble_type": "method",
    "models_used": ["model1", "model2"]
}}))
```

Generate only the code without explanations.
"""


class EnsemblePhase:
    """Ensemble Phase"""
    
    def __init__(self, llm: "LLMManager", executor: "CodeExecutor", memory: "AgentMemory"):
        self.llm = llm
        self.executor = executor
        self.memory = memory
    
    def run(self, best_models: list["ModelRecord"]) -> dict:
        context = self.memory.competition_context
        eda = self.memory.eda_insights
        
        if len(best_models) >= 3:
            ensemble_type = "stacking"
            strategy = "3+ models available, use stacking"
        elif len(best_models) >= 2:
            ensemble_type = "blending"
            strategy = "2 models, use weighted averaging with optimization"
        else:
            ensemble_type = "single"
            strategy = "Single model only"
        
        models_info = "\n".join([
            f"- {m.name}: {m.model_type}, CV={m.cv_score:.5f}"
            for m in best_models
        ]) if best_models else "No models"
        
        prompt = ENSEMBLE_PROMPT.format(
            task_type=context.get("task_type", "classification"),
            metric=context.get("metric", "auc"),
            target_column=context.get("target_column", "target"),
            data_dir=context.get("data_dir", "."),
            available_models=models_info,
            ensemble_strategy=strategy,
            ensemble_type=ensemble_type
        )
        
        code = self.llm.generate_code(prompt)
        result = self.executor.execute_with_retry(code, self.llm)
        
        if result.success:
            ensemble_result = self._parse_output(result.stdout)
            
            self.memory.add_experiment(
                phase="ensemble",
                description=f"Ensemble ({ensemble_result.get('ensemble_type', ensemble_type)})",
                code=code,
                cv_score=ensemble_result.get("cv_score", 0),
                success=True
            )
            return ensemble_result
        else:
            self.memory.add_experiment(
                phase="ensemble",
                description="Ensemble (failed)",
                code=code,
                success=False
            )
            best_score = best_models[0].cv_score if best_models else 0
            return {"cv_score": best_score, "ensemble_type": "failed"}
    
    def _parse_output(self, stdout: str) -> dict:
        lines = stdout.strip().split("\n")
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return {"cv_score": 0, "ensemble_type": "unknown"}
