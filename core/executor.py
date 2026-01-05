"""Code Executor - LLMが生成したPythonコードを安全に実行"""

import subprocess
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import traceback


@dataclass
class ExecutionResult:
    """コード実行結果"""
    success: bool
    stdout: str
    stderr: str
    return_value: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class CodeExecutor:
    """Pythonコードを安全に実行するクラス"""
    
    def __init__(
        self,
        working_dir: Path,
        timeout: int = 300,
        python_path: str = "python"
    ):
        self.working_dir = Path(working_dir)
        self.timeout = timeout
        self.python_path = python_path
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, code: str, save_script: bool = True) -> ExecutionResult:
        """コードを実行"""
        script_path = self.working_dir / "_agent_script.py"
        
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            result = subprocess.run(
                [self.python_path, str(script_path)],
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                error_type="RuntimeError" if result.returncode != 0 else None,
                error_message=result.stderr if result.returncode != 0 else None
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {self.timeout} seconds",
                error_type="TimeoutError",
                error_message=f"Code execution exceeded {self.timeout}s limit"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=traceback.format_exc(),
                error_type=type(e).__name__,
                error_message=str(e)
            )
        finally:
            if not save_script and script_path.exists():
                script_path.unlink()
    
    def execute_with_retry(
        self,
        code: str,
        llm_manager,
        max_retries: int = 3
    ) -> ExecutionResult:
        """エラー時にLLMで修正して再実行"""
        current_code = code
        
        for attempt in range(max_retries):
            result = self.execute(current_code)
            
            if result.success:
                return result
            
            if attempt < max_retries - 1:
                fix_prompt = f"""The following Python code has an error. Please fix it.

## Original Code:
```python
{current_code}
```

## Error:
{result.error_type}: {result.error_message}

{result.stderr}

Please provide the corrected code only, without explanations."""
                
                current_code = llm_manager.generate_code(fix_prompt)
                print(f"  Retry {attempt + 1}: Attempting to fix error...")
        
        return result
