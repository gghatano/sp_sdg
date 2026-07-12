# SPEC.md
# 信号データ拡張・論文追試プロジェクト仕様

## 0. 文書情報

- 対象: 公開信号データを用いたデータ拡張研究
- 初期対象: UCR Time Series Classification Archive
- 最終目的: 所定性能に必要な実被験者数の削減可能性を評価する
- 実行環境: Claude Code
- サブエージェント同時実行上限: 3
- レポート形式: Tailwind CSSによる静的HTML
- 初期化スクリプト: `init_project_structure.sh`

---

## 1. プロジェクト初期化

### 1.1 実行方法

```bash
chmod +x init_project_structure.sh
./init_project_structure.sh signal-augmentation-reproduction
cd signal-augmentation-reproduction
```

引数を省略した場合、既定のプロジェクト名は次のとおりとする。

```text
signal-augmentation-reproduction
```

### 1.2 初期化スクリプトの要件

`init_project_structure.sh`は以下を満たすこと。

- Bashで実行可能
- `set -Eeuo pipefail`を使用
- 冪等である
- 既存ファイルを上書きしない
- 既存ディレクトリを削除しない
- 空ディレクトリの追跡用に`.gitkeep`を配置する
- プロジェクトルートとして`/`を指定した場合は停止する
- 実行後に作成した上位ディレクトリを表示する
- 外部コマンドへの依存を最小限とする
- macOSおよび一般的なLinux環境で動作する

### 1.3 `.gitkeep`の扱い

`.gitkeep`はGitの正式機能ではなく、空ディレクトリをリポジトリに含めるための慣習として使用する。

次の原則で扱う。

- 初期化時は管理対象ディレクトリへ配置する
- 実ファイルが追加された後も削除必須とはしない
- 成果物やデータそのものをGit管理しない場合でも、ディレクトリ構造は保持する
- `.gitignore`によりディレクトリ配下を除外する場合は、`.gitkeep`を明示的に除外対象外とする

例:

```gitignore
data/raw/*
!data/raw/.gitkeep

runs/logs/*
!runs/logs/.gitkeep
```

---

## 2. プロジェクト構成

初期化スクリプトは以下の構造を作成する。

```text
signal-augmentation-reproduction/
├── .claude/
│   ├── agents/
│   │   └── .gitkeep
│   └── commands/
│       └── .gitkeep
├── config/
│   └── experiments/
│       └── .gitkeep
├── references/
│   ├── papers/
│   │   └── .gitkeep
│   └── notes/
│       └── .gitkeep
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── metadata/
│       └── .gitkeep
├── src/
│   └── signal_aug/
│       ├── data/
│       │   └── .gitkeep
│       ├── augmentations/
│       │   └── .gitkeep
│       ├── features/
│       │   └── .gitkeep
│       ├── models/
│       │   └── .gitkeep
│       ├── experiments/
│       │   └── .gitkeep
│       ├── evaluation/
│       │   └── .gitkeep
│       └── reporting/
│           └── .gitkeep
├── tests/
│   ├── unit/
│   │   └── .gitkeep
│   ├── integration/
│   │   └── .gitkeep
│   ├── regression/
│   │   └── .gitkeep
│   └── fixtures/
│       └── .gitkeep
├── scripts/
│   └── .gitkeep
├── runs/
│   ├── manifests/
│   │   └── .gitkeep
│   ├── checkpoints/
│   │   └── .gitkeep
│   ├── metrics/
│   │   └── .gitkeep
│   ├── predictions/
│   │   └── .gitkeep
│   ├── synthetic_samples/
│   │   └── .gitkeep
│   └── logs/
│       └── .gitkeep
├── artifacts/
│   └── .gitkeep
└── report/
    ├── src/
    │   └── .gitkeep
    ├── assets/
    │   ├── charts/
    │   │   └── .gitkeep
    │   └── data/
    │       └── .gitkeep
    └── dist/
        └── assets/
            └── .gitkeep
```

ルート直下の設定ファイル、Markdown、Pythonモジュール、HTMLファイルは、Claude CodeがPhase 0で生成する。

---

## 3. ディレクトリ責務

### 3.1 `.claude/`

Claude Code固有の設定を格納する。

```text
.claude/
├── agents/
└── commands/
```

`agents/`には最大3並列運用を前提としたサブエージェント定義を配置する。

想定エージェント:

- `research-analyst.md`
- `implementation-engineer.md`
- `result-auditor.md`
- `report-engineer.md`
- `experiment-runner.md`

同時起動は最大3とし、常に全エージェントを起動する必要はない。

`commands/`には反復利用するClaude Codeコマンドを配置する。

想定例:

- `run-smoke.md`
- `run-phase1.md`
- `audit-results.md`
- `build-report.md`

### 3.2 `config/`

データセット、拡張手法、モデル、実験条件をコードから分離する。

Phase 0で作成する想定ファイル:

```text
config/
├── datasets.yaml
├── augmentations.yaml
├── models.yaml
└── experiments/
    ├── smoke.yaml
    ├── phase1.yaml
    ├── phase2.yaml
    └── subject_count.yaml
```

コード中へ実験条件を直接埋め込まない。

### 3.3 `references/`

論文、補足資料、研究メモ、追試条件を格納する。

```text
references/
├── papers/
├── notes/
├── paper_matrix.csv
└── reproduction_targets.yaml
```

- `papers/`: 利用条件上保存可能な論文・補足資料
- `notes/`: 論文読解、実装差、再現条件
- `paper_matrix.csv`: 論文横断比較
- `reproduction_targets.yaml`: 追試対象の構造化定義

### 3.4 `data/`

データ処理段階を分離する。

- `raw/`: 取得した原データ。原則変更禁止
- `interim/`: 中間変換、キャッシュ
- `processed/`: 学習用に整形したデータ
- `metadata/`: checksum、ライセンス、データ辞書、分割情報

原データと加工データを同一場所へ保存しない。

### 3.5 `src/signal_aug/`

研究コード本体を格納する。

- `data/`: ローダ、分割、前処理
- `augmentations/`: 拡張手法
- `features/`: 統計、周波数、Wavelet特徴
- `models/`: 1D-CNN、MiniRocket、比較モデル
- `experiments/`: runner、seed、resume、追跡
- `evaluation/`: 指標、統計、信号品質、監査
- `reporting/`: 集計、図表、HTML入力データ生成

Phase 0で各パッケージへ`__init__.py`を追加する。

### 3.6 `tests/`

- `unit/`: 単一関数・クラス
- `integration/`: データ取得からmetrics保存まで
- `regression/`: 既知結果、schema、HTML構成
- `fixtures/`: 小規模固定データ

最低限、次を自動検査する。

- shape
- dtype
- seed再現性
- NaN / Inf
- train/test重複
- testデータへのfit
- metrics範囲
- run manifest schema
- HTML必須section

### 3.7 `scripts/`

運用・実験補助スクリプトを配置する。

想定ファイル:

- `download_datasets.py`
- `run_experiment.py`
- `aggregate_results.py`
- `audit_results.py`
- `build_report.py`
- `validate_artifacts.py`
- `compact_logs.py`

ライブラリ本体のロジックを`script`へ重複実装しない。

### 3.8 `runs/`

実験実行結果を格納する。

- `manifests/`: 条件、環境、Git commit、status
- `checkpoints/`: best modelおよび再開情報
- `metrics/`: JSON、CSV等の評価値
- `predictions/`: test予測
- `synthetic_samples/`: 代表合成信号または再生成情報
- `logs/`: 実行ログ

各runは一意な`run_id`を持つ。

同じ`run_id`へ複数プロセスが書き込んではならない。

### 3.9 `artifacts/`

研究進行と判断を記録する。

Phase 0で作成する想定ファイル:

- `state.md`
- `task_queue.yaml`
- `decision_log.md`
- `reproduction_notes.md`
- `deviations.md`
- `audit_checklist.md`
- `limitations.md`
- `findings.json`

`task_queue.yaml`を自律実行の唯一の順序管理簿とする。

### 3.10 `report/`

Tailwind CSSによる静的HTMLレポートを管理する。

```text
report/
├── package.json
├── tailwind.config.js
├── postcss.config.js
├── src/
│   ├── input.css
│   ├── report.template.html
│   └── report.js
├── assets/
│   ├── charts/
│   └── data/
└── dist/
    ├── index.html
    └── assets/
```

`dist/index.html`はオフラインで閲覧可能とする。

実験結果をHTMLへ手入力せず、`runs/`と`artifacts/`から自動生成する。

---

## 4. ルート直下に作成するファイル

初期化後、Claude CodeはPhase 0で以下を作成する。

```text
README.md
SPEC.md
RESEARCH_PLAN.md
CLAUDE.md
pyproject.toml
uv.lock
Makefile
.gitignore
.env.example
```

必要に応じて次も作成する。

```text
.python-version
.pre-commit-config.yaml
LICENSE
CITATION.cff
```

### 4.1 `README.md`

最低限含める内容:

- 目的
- 対象論文
- セットアップ
- smoke test
- Phase 1実行
- resume
- audit
- report生成
- ディレクトリ構成
- 制約

### 4.2 `CLAUDE.md`

最低限含める内容:

- プロジェクト目的
- Phase順序
- 最大3並列
- ディレクトリ責務
- testデータ利用禁止
- config駆動
- run manifest必須
- state/task queue更新義務
- 同一失敗の再試行上限
- HTMLへの手入力禁止
- 原論文追試と独自検証の区別

### 4.3 `.gitignore`

以下を考慮する。

- Python cache
- virtual environment
- Node modules
- 大規模データ
- checkpoints
- logs
- 生成HTMLの扱い
- `.gitkeep`の保持

`report/dist/index.html`を成果物としてGit管理するかは、運用方針に応じて決める。初期仕様ではGit管理可能とする。

---

## 5. 研究フェーズ

### Phase 0: 基盤構築

- プロジェクト初期化
- UCRローダ
- config
- runner
- seed
- manifest
- test
- 最小HTML
- サブエージェント定義

### Phase 1: UCR最小追試

対象:

- ECG5000
- FordA
- GunPoint

拡張候補:

- None
- Random oversampling
- Jittering
- Scaling
- Mixup
- DTW-based augmentation
- Raw-SMOTE

モデル:

- 1D-CNN
- MiniRocket + RidgeClassifier

反復:

- 3 seeds

### Phase 2: UCR横断比較

- 10データセット以上
- 学習サンプル比率10%、25%、50%、75%、100%
- 信号品質
- 統計検定
- 被験者データへ持ち込む3～5手法を選定

### Phase 3: 被験者ID付き公開データ選定

- 候補3件以上
- 主対象1件
- 予備対象1件
- group split
- ライセンス
- baseline
- データ辞書

### Phase 4: 被験者数学習曲線

- 被験者単位train/validation/test
- 合成なしbaseline
- 各人数で複数反復
- 目標性能事前登録
- 必要被験者数推定

### Phase 5: 被験者数削減評価

- 拡張手法別学習曲線
- 必要被験者数
- 削減率
- 等価実被験者数
- 信頼区間
- negative control

### Phase 6～7

- 手法改善
- 別データ検証
- 統合レポート
- 研究成果化

---

## 6. 最大3並列のサブエージェント

同時実行数は最大3とする。

基本レーン:

1. Research / Design
2. Implementation / Execution
3. Audit / Reporting

各サブエージェント依頼には以下を含める。

```text
Task ID:
Objective:
Inputs:
Allowed files:
Forbidden files:
Expected outputs:
Acceptance criteria:
Dependencies:
Return format:
```

返却形式:

```markdown
## Result
## Evidence
## Files changed
## Tests / commands
## Risks
## Unresolved
## Recommended next action
```

同一ファイル、同一run ID、同一集計ファイルを複数エージェントに同時編集させない。

---

## 7. 実験管理

各run manifestに最低限以下を保存する。

```json
{
  "run_id": "...",
  "phase": 1,
  "dataset": "...",
  "dataset_checksum": "...",
  "split_checksum": "...",
  "augmentation": "...",
  "augmentation_params": {},
  "model": "...",
  "model_params": {},
  "seed": 0,
  "git_commit": "...",
  "git_dirty": false,
  "python_version": "...",
  "package_lock_checksum": "...",
  "hardware": {},
  "started_at": "...",
  "ended_at": "...",
  "status": "completed",
  "metrics_path": "...",
  "predictions_path": "...",
  "log_path": "..."
}
```

同一条件の再実行時は、既存runのstatusと成果物を確認し、必要な場合だけ再開または再実行する。

---

## 8. データリーケージ防止

testデータを以下に利用してはならない。

- scaler、PCA、特徴抽出器のfit
- 拡張器のfit
- 合成元サンプル
- early stopping
- ハイパーパラメータ選択
- モデル選択
- 手法選択
- 目標性能の事後設定

被験者ID付きデータでは、同一被験者をtrainとtestへ分割してはならない。

UCRのサンプル数を被験者数として表現してはならない。

---

## 9. HTMLレポート

`report/dist/index.html`には以下を含める。

- Executive Summary
- 研究目的・仮説
- 現在のPhase
- 完了・実行中・未実施
- 論文追試条件
- データセット
- 拡張手法
- モデル
- 再現性情報
- 主要結果
- 原論文比較
- 失敗run
- 監査結果
- 限界
- 次のタスク

Phase 4以降では、実被験者数対性能の学習曲線を主要図とする。

Phase 5以降では、必要被験者数、削減率、信頼区間をExecutive Summaryへ掲載する。

---

## 10. 初回完了条件

初回のClaude Code実行では、最低限以下を完了する。

- 初期化スクリプトで構造作成
- Phase 0品質ゲート
- UCR 3データセット
- 5拡張手法以上
- 2モデル
- 3 seeds
- run manifest
- unit/integration test
- 結果監査
- Tailwind CSS HTML
- Phase 2のtask queue
- Phase 3候補調査タスクの登録

計画だけで停止せず、smoke test、最小追試、監査、HTML生成まで進める。

---

## 11. 停止条件

以下の場合は次Phaseへ進まず、現状と理由を記録する。

- データリーケージ
- testデータ利用
- ライセンス不明
- metric不整合
- split再現不能
- 同一障害が2回の修正後も継続
- リソース不足
- 研究目的とデータ構造の不一致

部分完了の場合も、`artifacts/state.md`、`task_queue.yaml`、HTMLレポートへ反映する。

