# Kaggle Tabular Agent

テーブルデータ向けKaggleコンペを自動で解くAIエージェント。

## 🚀 Google Colabでの実行方法

### 1. セットアップ

```python
# リポジトリをクローン
!git clone https://github.com/YOUR_USERNAME/kaggle-agent.git
%cd kaggle-agent

# 依存関係をインストール
!pip install -q llama-cpp-python pandas numpy lightgbm xgboost catboost scikit-learn

# llama-cpp-pythonをGPU対応でインストール（推奨）
!CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 2. モデルのダウンロード

```python
!pip install -q huggingface_hub

from huggingface_hub import hf_hub_download

# テキスト理解用モデル（軽量）
hf_hub_download(
    repo_id="unsloth/Llama-3.2-3B-Instruct-GGUF",
    filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    local_dir="./models"
)

# コード生成用モデル（MoE、L4で動作可能）
hf_hub_download(
    repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    filename="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    local_dir="./models"
)
```

### 3. Kaggleデータのダウンロード

```python
# Kaggle認証設定
from google.colab import files
files.upload()  # kaggle.json をアップロード

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# コンペデータをダウンロード（例: Titanic）
!kaggle competitions download -c titanic -p ./data/titanic
!unzip -o ./data/titanic/titanic.zip -d ./data/titanic
```

### 4. エージェント実行

```python
!python agent.py --competition ./data/titanic --max-iterations 5
```

または、Pythonから直接実行：

```python
from agent import KaggleTabularAgent, AgentConfig

config = AgentConfig(
    competition_dir="./data/titanic",
    text_model_path="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    code_model_path="models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    max_improvement_iterations=5
)

agent = KaggleTabularAgent(config)
result = agent.run()
print(f"Final Score: {result['final_score']}")
```

## 📋 必要なGPU

| モデル | 最小VRAM |
|-------|---------|
| Llama-3.2-3B (Q4) | ~3GB |
| Qwen3-Coder-30B-A3B (Q4) | ~18GB |

**推奨**: Colab Pro/Pro+ の **L4 GPU** (24GB)

> ⚠️ 無料版Colab (T4: 16GB) では Qwen3-Coder-30B-A3B が動作しない可能性があります。

## 📁 プロジェクト構造

```
kaggle-agent/
├── agent.py              # メインエージェント
├── requirements.txt
├── core/
│   ├── llm.py            # LLM管理
│   ├── executor.py       # コード実行
│   └── memory.py         # メモリ管理
├── phases/
│   ├── understanding.py  # コンペ理解
│   ├── eda.py            # EDA
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── ensemble.py
└── models/               # GGUFモデル配置
```

## 🔧 オプション

```bash
python agent.py \
  --competition ./data/titanic \
  --competition-name titanic \
  --text-model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --code-model ./models/Qwen3-Coder-30B-A3B-Q4_K_M.gguf \
  --max-iterations 10 \
  --target-score 0.85
```

| オプション | 説明 |
|-----------|------|
| `--competition` | データディレクトリのパス（必須） |
| `--competition-name` | Kaggleコンペ名（APIから情報取得） |
| `--max-iterations` | 改善ループの最大回数 |
| `--target-score` | 目標スコア（達成で終了） |

## 📝 出力

- `submission.csv` - Kaggle提出用ファイル
- `agent_result.json` - 実行結果のサマリー
# kaggle-agent
