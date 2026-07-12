---
description: 結果監査(機械検査 + 手動チェックリスト)を実施する
---

1. `make audit` を実行し、問題があれば run_id ごとに原因を調査
2. artifacts/audit_checklist.md の手動確認項目を検証(ベースライン相場・seed 分散・条件固定)
3. 監査結果を artifacts/state.md に記録し、問題は Implementation レーンへ差し戻す
