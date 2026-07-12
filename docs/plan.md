# タスク遂行計画(PLAN.md)

本書は [`docs/spec.md`](./spec.md) に定義された「信号データ拡張・論文追試プロジェクト」を遂行するための実行計画である。仕様の各要件をタスクへ分解し、順序・依存関係・成果物・完了条件を定める。

- 対象リポジトリ: `gghatano/sp_sdg`
- 仕様バージョン: `docs/spec.md`(commit `776e1e3`)
- 進行管理: `artifacts/task_queue.yaml` を唯一の順序管理簿とする(spec §3.9)
- 並列度: サブエージェント同時実行は最大 3(spec §0, §6)

---

## 1. 全体方針

1. **config 駆動**: データセット・拡張手法・モデル・実験条件はすべて `config/` の YAML に置き、コードへ埋め込まない(spec §3.2)。
2. **run manifest 必須**: すべての実験 run は一意な `run_id` と manifest(条件・環境・Git commit・status)を持つ(spec §7)。
3. **リーケージ防止を最優先**: test データの fit / 選択 / 合成元利用を禁止し、テストで自動検査する(spec §8)。
4. **レポートは自動生成**: HTML への手入力を禁止し、`runs/` と `artifacts/` から生成する(spec §3.10, §9)。
5. **計画だけで停止しない**: 初回実行では smoke test → 最小追試 → 監査 → HTML 生成まで到達する(spec §10)。
6. **停止条件の遵守**: spec §11 に該当する事象が発生した場合は次 Phase へ進まず、`artifacts/state.md` と HTML レポートへ現状と理由を記録する。

---

## 2. フェーズ別タスク分解

### Phase 0: 基盤構築

ゴール: 実験を再現可能に実行できる最小基盤を整備し、品質ゲートを通過する。

| # | タスク | 主な成果物 | 依存 |
|---|--------|-----------|------|
| 0-1 | ルート直下ファイル整備 | `README.md` / `CLAUDE.md` / `RESEARCH_PLAN.md` / `pyproject.toml` / `uv.lock` / `Makefile` / `.gitignore` / `.env.example` | なし |
| 0-2 | config 雛形作成 | `config/datasets.yaml` / `augmentations.yaml` / `models.yaml` / `experiments/{smoke,phase1,phase2,subject_count}.yaml` | 0-1 |
| 0-3 | パッケージ初期化 | `src/signal_aug/**/__init__.py` | 0-1 |
| 0-4 | UCR データローダ | `src/signal_aug/data/`(取得・checksum・分割・前処理)、`scripts/download_datasets.py` | 0-2, 0-3 |
| 0-5 | 実験 runner・seed 管理・manifest | `src/signal_aug/experiments/`、`scripts/run_experiment.py`(resume 対応) | 0-2, 0-3 |
| 0-6 | テスト基盤 | `tests/unit` / `integration` / `fixtures`(shape・dtype・seed 再現性・NaN/Inf・train/test 重複・manifest schema 検査) | 0-4, 0-5 |
| 0-7 | 最小 HTML レポート | `report/`(Tailwind ビルド一式、`dist/index.html` がオフライン閲覧可能) | 0-1 |
| 0-8 | サブエージェント・コマンド定義 | `.claude/agents/*.md`(5 種)、`.claude/commands/*.md`(`run-smoke` 等) | 0-1 |
| 0-9 | 進行管理ファイル初期化 | `artifacts/state.md` / `task_queue.yaml` / `decision_log.md` / `reproduction_notes.md` / `deviations.md` / `audit_checklist.md` / `limitations.md` / `findings.json` | 0-1 |

**品質ゲート(Phase 0 完了条件)**
- `make test`(unit + integration)が成功する
- smoke 実験(`config/experiments/smoke.yaml`)が manifest 付きで完走し、resume が機能する
- `report/dist/index.html` が生成され、必須 section を含む(regression テストで検査)

### Phase 1: UCR 最小追試

ゴール: ECG5000 / FordA / GunPoint × 拡張 7 条件 × 2 モデル × 3 seeds の格子を完走し、監査済みの結果を HTML に反映する。

| # | タスク | 内容 |
|---|--------|------|
| 1-1 | 拡張手法実装 | None / Random oversampling / Jittering / Scaling / Mixup / DTW-based / Raw-SMOTE(`src/signal_aug/augmentations/`、unit テスト付き) |
| 1-2 | モデル実装 | 1D-CNN、MiniRocket + RidgeClassifier(`src/signal_aug/models/`) |
| 1-3 | 実験実行 | 3 データセット × 7 拡張 × 2 モデル × 3 seeds(= 126 runs)を runner で実行 |
| 1-4 | 集計・監査 | `scripts/aggregate_results.py` / `audit_results.py`(metric 範囲・リーケージ・再現性の検査) |
| 1-5 | レポート反映 | 主要結果・失敗 run・監査結果を HTML へ自動反映 |

### Phase 2: UCR 横断比較

- 10 データセット以上へ拡大し、学習サンプル比率 10/25/50/75/100% の条件を追加
- 信号品質指標と統計検定を `src/signal_aug/evaluation/` に実装
- 被験者データへ持ち込む拡張手法 3〜5 件を選定し、`artifacts/decision_log.md` に根拠を記録

### Phase 3: 被験者 ID 付き公開データ選定

- 候補 3 件以上を調査し(ライセンス・被験者数・タスク適合性)、主対象 1 件・予備 1 件を決定
- group split(被験者単位分割)・データ辞書・baseline を整備

### Phase 4: 被験者数学習曲線

- 被験者単位の train/validation/test 分割で、合成なし baseline の学習曲線を作成
- 目標性能を**事前登録**した上で必要被験者数を推定(各人数で複数反復)

### Phase 5: 被験者数削減評価

- 拡張手法別の学習曲線から、必要被験者数・削減率・等価実被験者数・信頼区間を算出
- negative control を含め、Executive Summary に削減率と信頼区間を掲載

### Phase 6〜7: 改善・一般化・統合

- 手法改善、別データセットでの検証、統合レポート、研究成果化

---

## 3. 依存関係とクリティカルパス

```
Phase 0 (基盤) ──→ Phase 1 (最小追試) ──→ Phase 2 (横断比較) ──→ Phase 5 (削減評価)
                                              │                      ↑
                                              └──→ Phase 3 (データ選定) ──→ Phase 4 (学習曲線)
```

- Phase 3 の候補調査は Phase 2 と並行可能(Research レーンで先行着手する)
- クリティカルパスは Phase 0 → 1 → 2 → 4 → 5。Phase 0 の runner / manifest / テスト基盤が全体のボトルネックのため最優先で完成させる

---

## 4. サブエージェント運用(最大 3 並列)

spec §6 の 3 レーンで運用する。

| レーン | 担当エージェント | 主なタスク |
|--------|-----------------|-----------|
| Research / Design | `research-analyst` | 論文調査、`paper_matrix.csv`、Phase 3 候補調査 |
| Implementation / Execution | `implementation-engineer`, `experiment-runner` | 実装、実験実行 |
| Audit / Reporting | `result-auditor`, `report-engineer` | 結果監査、HTML 生成 |

- 依頼時は spec §6 の書式(Task ID / Objective / Inputs / Allowed files / …)を必ず使用する
- 同一ファイル・同一 run_id・同一集計ファイルの同時編集を禁止する

---

## 5. 品質・再現性の担保

- **テスト**: spec §3.6 の 9 項目(shape、dtype、seed 再現性、NaN/Inf、train/test 重複、test への fit、metrics 範囲、manifest schema、HTML 必須 section)を CI 相当の `make test` で常時検査
- **リーケージ防止**: spec §8 の禁止事項を evaluation/audit のコードとテスト両方で機械的に検査。UCR のサンプル数を被験者数として扱わない
- **再実行制御**: 同一条件の再実行時は既存 run の status を確認し、必要な場合のみ resume / 再実行(spec §7)
- **失敗時**: 同一障害が 2 回の修正後も継続する場合は停止し、`artifacts/state.md` に記録(spec §11)

---

## 6. 初回実行(初回完了条件)のチェックリスト

spec §10 に基づく。初回の Claude Code 実行では以下を最低限完了する。

- [ ] Phase 0 品質ゲート通過(テスト・smoke・最小 HTML)
- [ ] UCR 3 データセット(ECG5000 / FordA / GunPoint)取得・checksum 記録
- [ ] 拡張手法 5 種以上 + 2 モデル + 3 seeds の実験完走
- [ ] 全 run に manifest
- [ ] unit / integration テスト成功
- [ ] 結果監査の実施と記録
- [ ] Tailwind CSS HTML レポート生成
- [ ] Phase 2 の task queue 登録
- [ ] Phase 3 候補調査タスクの登録

---

## 7. リスクと対応

| リスク | 影響 | 対応 |
|--------|------|------|
| UCR データ取得先の変更・障害 | Phase 1 着手不能 | ミラー URL を `config/datasets.yaml` に複数記載、checksum 検証 |
| DTW ベース拡張の計算コスト | 実験時間の増大 | smoke 設定で小規模検証後にスケール、必要なら window 制約 |
| Phase 3 候補のライセンス不明 | 停止条件に該当 | 候補を 3 件以上確保し、ライセンス確認を選定基準の最初に置く |
| 実験 run の中断 | 結果欠損 | manifest の status + resume 機構で再開可能にする |
| レポートと結果の乖離 | 信頼性低下 | HTML 手入力禁止、regression テストで必須 section を検査 |

---

## 8. 直近のアクション

1. 本計画のレビューと合意(GitHub Issue 上で行う)
2. Phase 0 タスク(0-1〜0-9)を `artifacts/task_queue.yaml` に登録して着手
3. Phase 0 品質ゲート通過後、Phase 1 の実験格子を実行
