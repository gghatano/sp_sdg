# audit_checklist.md

結果確定前に確認する項目。機械検査(scripts/audit_results.py)+ 手動確認。

本ファイルは Phase ごとに使い回す**恒久チェックリスト(テンプレート)**であり、チェックボックスは特定の run に紐付いた記録ではない。各 Phase の実際の機械検査結果は `artifacts/audit_report.json`(`make audit` が生成)と `artifacts/state.md` の各 Phase 記述を参照すること。

## 機械検査(make audit)

- [ ] 全 manifest が schema に適合(必須キー・status)
- [ ] metrics が [0, 1] の範囲内で NaN なし
- [ ] 予測ファイルの行数 = テストサンプル数
- [ ] 失敗 run の把握と原因記録

## 手動確認

- [ ] ベースライン(none)の精度が既知の相場と乖離していないか(例: GunPoint はほぼ 0.9 以上が相場)
- [ ] 拡張条件間でモデル・ハイパラが固定されているか(config の diff 確認)
- [ ] seed 間分散が異常に大きい条件はないか
- [ ] 「原論文追試」と「独自検証」の区別が記録されているか
- [ ] レポートの数値が results.json と一致しているか(手入力混入がないか)
