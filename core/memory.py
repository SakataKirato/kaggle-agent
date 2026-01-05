"""Agent Memory - 試行履歴とコンテキスト管理"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class Experiment:
    """1回の試行を記録"""
    timestamp: str
    phase: str
    description: str
    code: str
    cv_score: Optional[float] = None
    lb_score: Optional[float] = None
    success: bool = True
    notes: str = ""
    features_used: list[str] = field(default_factory=list)
    model_type: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "phase": self.phase,
            "description": self.description,
            "cv_score": self.cv_score,
            "lb_score": self.lb_score,
            "success": self.success,
            "notes": self.notes,
            "model_type": self.model_type
        }


@dataclass
class ModelRecord:
    """学習済みモデルの記録"""
    name: str
    model_type: str
    cv_score: float
    model_path: Optional[str] = None
    features: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)


class AgentMemory:
    """エージェントのメモリシステム"""
    
    def __init__(self):
        self.experiments: list[Experiment] = []
        self.competition_context: dict = {}
        self.best_score: float = 0.0
        self.best_models: list[ModelRecord] = []
        self.eda_insights: dict = {}
        self.feature_ideas: list[str] = []
        self.tried_features: set[str] = set()
    
    def set_competition_context(self, context: dict):
        self.competition_context = context
    
    def add_experiment(
        self,
        phase: str,
        description: str,
        code: str,
        cv_score: Optional[float] = None,
        success: bool = True,
        notes: str = "",
        model_type: Optional[str] = None
    ):
        exp = Experiment(
            timestamp=datetime.now().isoformat(),
            phase=phase,
            description=description,
            code=code,
            cv_score=cv_score,
            success=success,
            notes=notes,
            model_type=model_type
        )
        self.experiments.append(exp)
        
        if cv_score and cv_score > self.best_score:
            self.best_score = cv_score
    
    def update_best_score(self, score: float):
        if score > self.best_score:
            self.best_score = score
    
    def add_model(self, model: ModelRecord):
        self.best_models.append(model)
        self.best_models.sort(key=lambda m: m.cv_score, reverse=True)
        self.best_models = self.best_models[:5]
    
    def get_best_models(self, top_k: int = 5) -> list[ModelRecord]:
        return self.best_models[:top_k]
    
    def get_history(self) -> list[dict]:
        return [exp.to_dict() for exp in self.experiments]
    
    def get_history_summary(self) -> str:
        if not self.experiments:
            return "No experiments recorded yet."
        
        summary_lines = ["## Experiment History\n"]
        
        for i, exp in enumerate(self.experiments[-10:], 1):
            status = "✓" if exp.success else "✗"
            score = f"CV: {exp.cv_score:.5f}" if exp.cv_score else "N/A"
            summary_lines.append(
                f"{i}. [{status}] {exp.phase}: {exp.description} ({score})"
            )
        
        summary_lines.append(f"\nBest Score: {self.best_score:.5f}")
        summary_lines.append(f"Total Experiments: {len(self.experiments)}")
        
        return "\n".join(summary_lines)
    
    def add_feature_idea(self, idea: str):
        if idea not in self.tried_features:
            self.feature_ideas.append(idea)
    
    def mark_feature_tried(self, feature: str):
        self.tried_features.add(feature)
        if feature in self.feature_ideas:
            self.feature_ideas.remove(feature)
    
    def get_untried_features(self) -> list[str]:
        return [f for f in self.feature_ideas if f not in self.tried_features]
    
    def save(self, path: Path):
        data = {
            "competition_context": self.competition_context,
            "best_score": self.best_score,
            "experiments": [exp.to_dict() for exp in self.experiments],
            "eda_insights": self.eda_insights,
            "feature_ideas": self.feature_ideas,
            "tried_features": list(self.tried_features)
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: Path):
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self.competition_context = data.get("competition_context", {})
            self.best_score = data.get("best_score", 0.0)
            self.eda_insights = data.get("eda_insights", {})
            self.feature_ideas = data.get("feature_ideas", [])
            self.tried_features = set(data.get("tried_features", []))
