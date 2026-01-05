"""LLM Manager - 複数モデルを動的に切り替えて使用"""

from llama_cpp import Llama
from pathlib import Path
from typing import Optional
import gc


class LLMManager:
    """LLMの管理と切り替えを行うクラス"""
    
    def __init__(
        self,
        text_model_path: str = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        code_model_path: str = "models/Qwen3-Coder-30B-A3B-Q4_K_M.gguf",
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
    ):
        self.text_model_path = Path(text_model_path)
        self.code_model_path = Path(code_model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        
        self._text_model: Optional[Llama] = None
        self._code_model: Optional[Llama] = None
        self._current_model_type: Optional[str] = None
    
    def _load_text_model(self) -> Llama:
        """テキスト理解用モデルをロード"""
        if self._text_model is None:
            if self._code_model is not None:
                del self._code_model
                self._code_model = None
                gc.collect()
            
            print(f"Loading text model: {self.text_model_path}")
            self._text_model = Llama(
                model_path=str(self.text_model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            self._current_model_type = "text"
        return self._text_model
    
    def _load_code_model(self) -> Llama:
        """コード生成用モデルをロード"""
        if self._code_model is None:
            if self._text_model is not None:
                del self._text_model
                self._text_model = None
                gc.collect()
            
            print(f"Loading code model: {self.code_model_path}")
            self._code_model = Llama(
                model_path=str(self.code_model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            self._current_model_type = "code"
        return self._code_model
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """テキスト理解・生成（Llama-3.2-3B使用）"""
        model = self._load_text_model()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return response["choices"][0]["message"]["content"]
    
    def generate_code(
        self,
        prompt: str,
        system_prompt: str = "You are an expert Python programmer. Generate clean, efficient code.",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """コード生成（Qwen3-Coder-30B使用）"""
        model = self._load_code_model()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        content = response["choices"][0]["message"]["content"]
        return self._extract_code(content)
    
    def _extract_code(self, content: str) -> str:
        """レスポンスからPythonコードを抽出"""
        if "```python" in content:
            start = content.find("```python") + len("```python")
            end = content.find("```", start)
            if end != -1:
                return content[start:end].strip()
        
        if "```" in content:
            start = content.find("```") + 3
            newline = content.find("\n", start)
            if newline != -1:
                start = newline + 1
            end = content.find("```", start)
            if end != -1:
                return content[start:end].strip()
        
        return content.strip()
    
    def unload_all(self):
        """全モデルをメモリから解放"""
        if self._text_model is not None:
            del self._text_model
            self._text_model = None
        if self._code_model is not None:
            del self._code_model
            self._code_model = None
        gc.collect()
        self._current_model_type = None
