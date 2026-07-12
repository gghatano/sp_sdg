---
description: Phase 1 実験格子(UCR 3データセット x 7拡張 x 2モデル x 3seeds)を実行する
---

1. `make download` で UCR データと checksum を確認
2. `make phase1` を実行(resume 対応なので中断後も同じコマンドで再開可能)
3. 完了後 `make all-report` で監査・集計・レポート生成
4. 失敗 run があれば原因を分類し、同一失敗の再試行は 2 回まで
5. artifacts/task_queue.yaml と state.md を更新
