"""Understanding Phase - Kaggle APIを使ってコンペ情報を取得"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional
import json
import os

if TYPE_CHECKING:
    from core.llm import LLMManager
    from core.memory import AgentMemory


class UnderstandingPhase:
    """コンペ理解フェーズ（Kaggle API使用）"""
    
    def __init__(self, llm: "LLMManager", memory: "AgentMemory"):
        self.llm = llm
        self.memory = memory
        self._api = None
    
    def _get_kaggle_api(self):
        """Kaggle APIを初期化"""
        if self._api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self._api = KaggleApi()
            self._api.authenticate()
        return self._api
    
    def run(self, competition_dir: Path, competition_name: Optional[str] = None) -> dict:
        """コンペ情報を取得"""
        context = {}
        
        # Kaggle APIでコンペ情報を取得
        if competition_name:
            try:
                api_info = self._fetch_from_kaggle_api(competition_name)
                context.update(api_info)
                print(f"  Fetched competition info from Kaggle API")
            except Exception as e:
                print(f"  Warning: Could not fetch from Kaggle API: {e}")
        
        # ローカルファイルからも情報を補完
        local_info = self._analyze_local_data(competition_dir)
        
        # APIで取得できなかった情報をローカルから補完
        for key, value in local_info.items():
            if key not in context or context[key] is None:
                context[key] = value
        
        context["data_dir"] = str(competition_dir)
        context["available_files"] = self._list_data_files(competition_dir)
        
        return context
    
    def _fetch_from_kaggle_api(self, competition_name: str) -> dict:
        """Kaggle APIからコンペ情報を取得"""
        api = self._get_kaggle_api()
        
        # コンペ情報を取得
        competitions = api.competitions_list(search=competition_name)
        
        if not competitions:
            raise ValueError(f"Competition '{competition_name}' not found")
        
        # 最も一致するコンペを選択
        comp = None
        for c in competitions:
            if c.ref == competition_name or competition_name in c.ref:
                comp = c
                break
        
        if comp is None:
            comp = competitions[0]
        
        # 評価指標からタスクタイプを推測
        metric = comp.evaluationMetric or "unknown"
        task_type = self._infer_task_type(metric)
        
        return {
            "competition_name": comp.ref,
            "title": comp.title,
            "description": comp.description or "",
            "metric": metric,
            "task_type": task_type,
            "deadline": str(comp.deadline) if comp.deadline else None,
            "category": comp.category or "unknown",
            "reward": comp.reward or "unknown",
        }
    
    def _infer_task_type(self, metric: str) -> str:
        """評価指標からタスクタイプを推測"""
        metric_lower = metric.lower()
        
        classification_metrics = [
            "auc", "accuracy", "f1", "logloss", "log loss", 
            "precision", "recall", "mcc", "kappa"
        ]
        
        regression_metrics = [
            "rmse", "mse", "mae", "mape", "rmsle", "r2"
        ]
        
        for m in classification_metrics:
            if m in metric_lower:
                return "classification"
        
        for m in regression_metrics:
            if m in metric_lower:
                return "regression"
        
        return "unknown"
    
    def _analyze_local_data(self, competition_dir: Path) -> dict:
        """ローカルデータを分析"""
        info = {
            "target_column": None,
            "num_samples": 0,
            "num_features": 0
        }
        
        # train.csvがあれば分析
        train_path = competition_dir / "train.csv"
        if train_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(train_path, nrows=100)
                
                info["num_samples"] = len(pd.read_csv(train_path, usecols=[0]))
                info["num_features"] = len(df.columns)
                
                # 目的変数を推測
                for col in ["target", "Target", "label", "Label", "y", "class"]:
                    if col in df.columns:
                        info["target_column"] = col
                        break
                
                # 最後のカラムが目的変数の可能性
                if info["target_column"] is None:
                    info["target_column"] = df.columns[-1]
                    
            except Exception:
                pass
        
        return info
    
    def _list_data_files(self, competition_dir: Path) -> list[str]:
        """データファイル一覧を取得"""
        files = []
        for ext in ["*.csv", "*.parquet", "*.feather"]:
            files.extend([f.name for f in competition_dir.glob(ext)])
        return files
    
    def download_competition_data(self, competition_name: str, output_dir: Path) -> Path:
        """コンペデータをダウンロード"""
        api = self._get_kaggle_api()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Downloading competition data: {competition_name}")
        api.competition_download_files(competition_name, path=str(output_dir), quiet=False)
        
        # ZIPファイルを解凍
        import zipfile
        for zip_file in output_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(output_dir)
            zip_file.unlink()  # ZIPを削除
        
        print(f"  Downloaded to: {output_dir}")
        return output_dir
