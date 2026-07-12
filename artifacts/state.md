# state.md

最終更新: 2026-07-12(Phase 1 完了)

## 現在の状態

- Phase 0 完了(品質ゲート通過、テスト 51 件全通過)
- **Phase 1 完了**: 126 runs(3データセット × 7拡張 × 2モデル × 3 seeds)全完走・失敗 0・監査 130/130 合格
- 主要知見は artifacts/findings.json(F-1〜F-4)。フルサイズ学習データでは改善は +1pt 未満・悪化は最大 -3.5pt、最小データセット GunPoint でのみ改善傾向(ノイズ範囲)、FordA+MiniRocket では拡張が有害
- ペルソナレビュー(教授/院生/実務者の 3 視点)完了。指摘 24 件中、findings の主張修正・レポートの両側提示・task_queue の YAML 破損・再現性情報のバグ(commit 選択、git_dirty 判定)・README 補強など主要指摘を反映済み
- 次: Phase 2 準備(P2-1: データセット選定、P2-2: 学習サンプル比率スイープの runner 拡張)

## Phase 2 実行メモ(2026-07-12)

- Phase 2 グリッド初回実行は、二重起動した 2 プロセスが同一 run_id を並行書き込みして競合し、manifest 保存の tmp.replace が FileNotFoundError で約 183 runs 時点でクラッシュした
- 対策済み: runner に pid ロック(runs/.runner.lock)を追加し二重起動を拒否、manifest の tmp 名を pid 付きに変更。テストで担保(test_grid_lock.py)
- 単一プロセスで resume 実行中。resume は completed をスキップし、running/failed/未実行を再実行する

## 環境

- Python 3.11 / uv 管理(torch CPU 版・aeon 1.5)
- 実行環境: Claude Code リモートコンテナ(CPU のみ)
- Phase 1 の実測実行時間: 約 1.5 時間(FordA の CNN が支配的)

## 注意事項

- report/dist/index.html は Git 管理する方針(ユーザー決定 2026-07-12)
- LICENSE の選定は保留中(ユーザー決定 2026-07-12)
- Phase 2 の学習サンプル比率スイープには runner の拡張(train_fractions 対応)が必要
