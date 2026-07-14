# state.md

最終更新: 2026-07-13(issue #13 レポート体裁ブラッシュアップ)

## 現在の状態

- **Phase 0〜5 完了**。全 2797 runs・失敗 0・監査全合格。テストは `make test` で全通過(件数は増減するためここには固定値を書かない)
- PR #1 が main にマージ済み。レポートは GitHub Pages に自動デプロイ(https://gghatano.github.io/sp_sdg/ )
- 主要知見は artifacts/findings.json(F-1〜F-10)
- **issue #13 対応(branch claude/issue-13-resolution-r7ow7w)**: 論文タブを学会誌構成(要旨/序論/問題設定/提案手法(評価枠組み)/関連手法/実験設計/実験結果/考察/限界/まとめ/参考文献)へ再編。傍論(進捗・再現性・簡易実験・用語説明)はダッシュボードタブへ分離。3ペルソナレビュー+結果監査を反映して収束(要旨の削減ロジック精緻化、negative control 箱の固定数値を束縛化、追試/独自の明示、用語インライン注)。数値の手入力なし・test 利用示唆なしを監査確認。build/section テスト green
- 次の候補: Phase 6-7(削減評価の反復数増による確度向上、予備対象 WISDM での再現、拡張強度スイープ、統合レポート)。詳細は GitHub issue #7

## フェーズ別の経過(時系列)

### Phase 0 完了
基盤構築(config 駆動・runner・manifest・監査・Tailwind HTML 自動生成)。品質ゲート(smoke)通過。

### Phase 1 完了
126 runs(3データセット × 7拡張 × 2モデル × 3 seeds)全完走・監査 130/130 合格。知見 F-1〜F-4: フルサイズ学習データでは改善は +1pt 未満・悪化は最大 -3.5pt、GunPoint でのみ改善傾向(ノイズ範囲)、FordA+MiniRocket では拡張が有害。ベースラインは文献相場と整合。3ペルソナレビューを反映。

### Phase 2 完了
12データセット × 5学習比率 × 7拡張 × 2モデル × 3 seeds = 2520 runs 全完走・監査合格。知見 F-5(1D-CNN では拡張が Holm 補正後も有意)、F-6(MiniRocket ではどの手法も非有意)、F-7(効果は低〜中比率で相対的に大)。
- 初回実行は runner の二重起動で同一 run_id を並行書き込みし競合クラッシュ(約183 runs 時点)。対策: pid ロック(runs/.runner.lock)で二重起動拒否 + manifest の tmp 名を pid 付きに変更(test_grid_lock.py で担保)。単一プロセスで resume し完走。

### Phase 3 完了
被験者ID付き公開データを調査(全 CC BY 4.0)。主対象 UCI HAR(30名)、予備 WISDM(51名)を決定。

### Phase 4-5 完了
UCI HAR で被験者数学習曲線 126 runs + negative control 21 runs 完走・監査合格。事前登録どおり target=0.90 を結果前に固定。知見 F-8/F-9/F-10: 目標 0.90 に必要な実被験者数は none で約 8.85 名。点推定では DTW が最大削減(N*=7.9, 10.8%, 約1名節約)だが、negative control(label_shuffle)ですら約 4.3% の見かけの削減を示すため、約 4% 以下の削減はアーティファクトと区別できず、**本設定でデータ拡張が被験者数を確実に削減するとは言えない**。反復 3 回で CI が広い。

## 環境

- Python 3.11 / uv 管理(torch CPU 版・aeon 1.5)
- 実行環境: Claude Code リモートコンテナ(CPU のみ)
- 実測実行時間の目安: Phase 1 約 1.5 時間、Phase 2 約 2 時間、Phase 4-5 は cnn1d_har(軽量)で数十分〜

## 注意事項

- report/dist/index.html は Git 管理する方針(ユーザー決定 2026-07-12)
- LICENSE の選定は保留中(ユーザー決定 2026-07-12)
- レビュー指摘は GitHub issue #3(コード)/#4(テスト)/#5(ドキュメント)/#6(統計)に整理済み
