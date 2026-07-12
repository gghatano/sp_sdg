---
description: smoke 実験(Phase 0 品質ゲート)を実行して結果を確認する
---

以下を順に実行し、すべて成功することを確認してください。

1. `make test`
2. `make smoke`
3. `make all-report`

失敗した場合は logs(runs/logs/)を確認して原因を報告してください。成功したら artifacts/task_queue.yaml の該当タスクを更新してください。
