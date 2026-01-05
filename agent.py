# Kaggle Tabular Agent
from core.llm import LLMManager
from core.executor import CodeExecutor
from core.memory import AgentMemory
from phases.understanding import UnderstandingPhase
from phases.eda import EDAPhase
from phases.feature_engineering import FeatureEngineeringPhase
from phases.modeling import ModelingPhase
from phases.ensemble import EnsemblePhase

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class AgentConfig:
    """エージェント設定"""
    competition_dir: str
    competition_name: Optional[str] = None  # Kaggle competition name (e.g., 'titanic')
    text_model_path: str = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    code_model_path: str = "models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
    max_improvement_iterations: int = 10
    target_score: Optional[float] = None


class KaggleTabularAgent:
    """Kaggleテーブルデータコンペ向け自律エージェント"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.competition_dir = Path(config.competition_dir)
        
        # Core components
        self.llm = LLMManager(
            text_model_path=config.text_model_path,
            code_model_path=config.code_model_path
        )
        self.executor = CodeExecutor(working_dir=self.competition_dir)
        self.memory = AgentMemory()
        
        # Phase modules
        self.understanding = UnderstandingPhase(self.llm, self.memory)
        self.eda = EDAPhase(self.llm, self.executor, self.memory)
        self.feature_engineering = FeatureEngineeringPhase(self.llm, self.executor, self.memory)
        self.modeling = ModelingPhase(self.llm, self.executor, self.memory)
        self.ensemble = EnsemblePhase(self.llm, self.executor, self.memory)
    
    def run(self) -> dict:
        """エージェントを実行"""
        print("=" * 60)
        print("Kaggle Tabular Agent Starting...")
        print("=" * 60)
        
        # 1. コンペ理解
        print("\n[Phase 1/5] Understanding Competition...")
        context = self.understanding.run(
            self.competition_dir,
            competition_name=self.config.competition_name
        )
        self.memory.set_competition_context(context)
        print(f"  - Competition: {context.get('competition_name', 'local')}")
        print(f"  - Metric: {context.get('metric', 'unknown')}")
        print(f"  - Task: {context.get('task_type', 'unknown')}")
        
        # 2. EDA
        print("\n[Phase 2/5] Exploratory Data Analysis...")
        eda_insights = self.eda.run(context)
        print(f"  - Features: {eda_insights.get('num_features', 0)}")
        print(f"  - Samples: {eda_insights.get('num_samples', 0)}")
        
        # 3. ベースライン作成
        print("\n[Phase 3/5] Creating Baseline...")
        baseline_result = self.modeling.create_baseline(eda_insights)
        best_score = baseline_result.get("cv_score", 0)
        print(f"  - Baseline CV Score: {best_score:.5f}")
        
        # 4. 改善ループ
        print("\n[Phase 4/5] Improvement Loop...")
        for i in range(self.config.max_improvement_iterations):
            print(f"\n  Iteration {i+1}/{self.config.max_improvement_iterations}")
            
            # 特徴量エンジニアリング
            fe_result = self.feature_engineering.run(eda_insights, self.memory.get_history())
            
            # モデル改善
            model_result = self.modeling.improve(fe_result)
            new_score = model_result.get("cv_score", 0)
            
            if new_score > best_score:
                improvement = new_score - best_score
                print(f"    ✓ Improved: {best_score:.5f} -> {new_score:.5f} (+{improvement:.5f})")
                best_score = new_score
                self.memory.update_best_score(best_score)
            else:
                print(f"    ✗ No improvement: {new_score:.5f}")
            
            if self.config.target_score and best_score >= self.config.target_score:
                print(f"\n  🎯 Target score reached!")
                break
        
        # 5. アンサンブル
        print("\n[Phase 5/5] Ensemble...")
        ensemble_result = self.ensemble.run(self.memory.get_best_models())
        final_score = ensemble_result.get("cv_score", best_score)
        print(f"  - Final CV Score: {final_score:.5f}")
        
        result = {
            "final_score": final_score,
            "iterations": i + 1,
            "experiments": len(self.memory.experiments)
        }
        
        print("\n" + "=" * 60)
        print(f"Agent Completed! Final Score: {final_score:.5f}")
        print("=" * 60)
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kaggle Tabular Agent")
    parser.add_argument("--competition", required=True, help="Path to competition directory")
    parser.add_argument("--competition-name", default=None, help="Kaggle competition name (e.g., 'titanic')")
    parser.add_argument("--text-model", default="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--code-model", default="models/Qwen3-Coder-30B-A3B-Q4_K_M.gguf")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--target-score", type=float, default=None)
    
    args = parser.parse_args()
    
    config = AgentConfig(
        competition_dir=args.competition,
        competition_name=args.competition_name,
        text_model_path=args.text_model,
        code_model_path=args.code_model,
        max_improvement_iterations=args.max_iterations,
        target_score=args.target_score
    )
    
    agent = KaggleTabularAgent(config)
    result = agent.run()
    
    output_path = Path(args.competition) / "agent_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()
