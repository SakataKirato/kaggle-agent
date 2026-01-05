"""Feature Engineering Phase - Generate new features"""

from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from core.llm import LLMManager
    from core.executor import CodeExecutor
    from core.memory import AgentMemory


FE_CODE_PROMPT = """You are a Kaggle Grandmaster. Generate Python code for feature engineering based on the following info.

## Competition Info:
- Task: {task_type}
- Target column: {target_column}
- Data directory: {data_dir}

## EDA Results:
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

## Previous Experiments:
{experiment_history}

## Features to Generate (pick 1-3):
{feature_ideas}

## Requirements:
1. Load train.csv (or train_fe.csv if exists)
2. Generate new features
3. Save as train_fe.csv
4. Apply the same transformations to test.csv and save as test_fe.csv
5. Output results in JSON format

## Output Format:
```
print(json.dumps({{
    "new_features": ["feature_name1", "feature_name2"],
    "description": "what was done"
}}))
```

Generate only the code without explanations.
"""


class FeatureEngineeringPhase:
    """Feature Engineering Phase"""
    
    def __init__(self, llm: "LLMManager", executor: "CodeExecutor", memory: "AgentMemory"):
        self.llm = llm
        self.executor = executor
        self.memory = memory
    
    def run(self, eda_insights: dict, history: list[dict]) -> dict:
        context = self.memory.competition_context
        
        untried = self.memory.get_untried_features()
        if not untried:
            untried = self._generate_feature_ideas(eda_insights)
        
        prompt = FE_CODE_PROMPT.format(
            task_type=context.get("task_type", "unknown"),
            target_column=context.get("target_column", "target"),
            data_dir=context.get("data_dir", "."),
            numeric_columns=eda_insights.get("numeric_columns", []),
            categorical_columns=eda_insights.get("categorical_columns", []),
            experiment_history=self._format_history(history),
            feature_ideas="\n".join(f"- {idea}" for idea in untried[:5])
        )
        
        code = self.llm.generate_code(prompt)
        result = self.executor.execute_with_retry(code, self.llm)
        
        if result.success:
            fe_result = self._parse_output(result.stdout)
            
            for idea in untried[:3]:
                self.memory.mark_feature_tried(idea)
            
            self.memory.add_experiment(
                phase="feature_engineering",
                description=fe_result.get("description", "FE"),
                code=code,
                success=True
            )
            return fe_result
        else:
            self.memory.add_experiment(
                phase="feature_engineering",
                description="FE (failed)",
                code=code,
                success=False
            )
            return {"new_features": [], "description": "Failed"}
    
    def _generate_feature_ideas(self, eda_insights: dict) -> list[str]:
        ideas = []
        numeric = eda_insights.get("numeric_columns", [])
        categorical = eda_insights.get("categorical_columns", [])
        
        if len(numeric) >= 2:
            ideas.append(f"Ratio between {numeric[0]} and {numeric[1]}")
        if categorical:
            ideas.append(f"Target encoding for {categorical[0]}")
        if numeric:
            ideas.append(f"Log transform of {numeric[0]}")
        
        for idea in ideas:
            self.memory.add_feature_idea(idea)
        
        return ideas
    
    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return "No previous experiments."
        return "\n".join([f"- {exp.get('description', 'N/A')}" for exp in history[-5:]])
    
    def _parse_output(self, stdout: str) -> dict:
        lines = stdout.strip().split("\n")
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return {"new_features": [], "description": "Unknown"}
