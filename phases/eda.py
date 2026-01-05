"""EDA Phase - Exploratory Data Analysis"""

from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from core.llm import LLMManager
    from core.executor import CodeExecutor
    from core.memory import AgentMemory


EDA_CODE_PROMPT = """You are a data scientist. Generate Python code for EDA based on the following competition info.

## Competition Info:
- Task: {task_type}
- Metric: {metric}
- Target column: {target_column}
- Data directory: {data_dir}
- Files: {available_files}

## Requirements:
1. Load train.csv
2. Calculate basic statistics
3. Check missing values
4. Check target distribution
5. Output results in JSON format

## Output Format:
At the end of the code, print a JSON object:
```
print(json.dumps({{
    "num_samples": number_of_rows,
    "num_features": number_of_features,
    "missing_columns": [columns_with_missing],
    "numeric_columns": [numeric_columns],
    "categorical_columns": [categorical_columns],
    "insights": ["insight1", "insight2"]
}}))
```

Generate only the code without explanations.
"""


class EDAPhase:
    """EDA Phase (uses Qwen3-Coder)"""
    
    def __init__(self, llm: "LLMManager", executor: "CodeExecutor", memory: "AgentMemory"):
        self.llm = llm
        self.executor = executor
        self.memory = memory
    
    def run(self, context: dict) -> dict:
        prompt = EDA_CODE_PROMPT.format(
            task_type=context.get("task_type", "unknown"),
            metric=context.get("metric", "unknown"),
            target_column=context.get("target_column", "target"),
            data_dir=context.get("data_dir", "."),
            available_files=context.get("available_files", [])
        )
        
        code = self.llm.generate_code(prompt)
        result = self.executor.execute_with_retry(code, self.llm)
        
        if result.success:
            insights = self._parse_eda_output(result.stdout)
            
            self.memory.add_experiment(
                phase="eda",
                description="Initial EDA",
                code=code,
                success=True,
                notes=f"Found {insights.get('num_features', 0)} features"
            )
            self.memory.eda_insights = insights
            return insights
        else:
            self.memory.add_experiment(
                phase="eda",
                description="Initial EDA (failed)",
                code=code,
                success=False,
                notes=result.error_message or "Unknown error"
            )
            return self._default_insights()
    
    def _parse_eda_output(self, stdout: str) -> dict:
        lines = stdout.strip().split("\n")
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return self._default_insights()
    
    def _default_insights(self) -> dict:
        return {
            "num_samples": 0,
            "num_features": 0,
            "missing_columns": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "insights": []
        }
