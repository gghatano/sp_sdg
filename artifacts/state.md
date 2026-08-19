# state.md

最終更新: 2026-07-20(issue #21 §6.7: 4データ統合まとめ追加・DS-5 は deferred 確定 DS5-1/2)

## 現在の状態

- **Phase 0〜5 完了 + Phase 6 の WISDM 外的妥当性(issue #21 DS-1)完了**。全 2965 runs・失敗 0・監査全合格。テストは `make test` で全通過(件数は増減するためここには固定値を書かない)
- PR #1 が main にマージ済み。レポートは GitHub Pages に自動デプロイ(https://gghatano.github.io/sp_sdg/ )
- 主要知見は artifacts/findings.json(F-1〜F-12)
- **issue #23 対応(レッドチーム検証の文書・解釈是正)**: F-12/§6.4 の二次解釈を是正。(1) label_shuffle を「クラス信号ゼロの床」から「元データ保持+ラベルノイズの悲観的対照」に訂正し、純量水増しの基準は oversample に統一(F-10 にも波及)。(2) F-12 の「F-8/F-10 を再現」を「再現したのは F-8 の帰無のみ、F-10 の核は WISDM で非再現」に格下げ。(3) smote +10.5% は accuracy 固有で macro-F1 では順位反転(smote のみ僅かに正、dtw 最下位。いずれも CI 重複)を明記。(4) 引用を weiss2019(2019, 51名)から kwapisz2011(v1.1, 36名)へ差し替え。(5) UCR データセット数を distinct 13 と被験者データ 2 に分離(二重計上是正)。(6) none 曲線の非単調性・target=0.80 のプラトー肩・前処理非対称(追試 framing)を明記。数値は results.json / 統括検算値と整合、HTML 手入力なし
- **issue #13 対応(branch claude/issue-13-resolution-r7ow7w)**: 論文タブを学会誌構成(要旨/序論/問題設定/提案手法(評価枠組み)/関連手法/実験設計/実験結果/考察/限界/まとめ/参考文献)へ再編。傍論(進捗・再現性・簡易実験・用語説明)はダッシュボードタブへ分離。3ペルソナレビュー+結果監査を反映して収束(要旨の削減ロジック精緻化、negative control 箱の固定数値を束縛化、追試/独自の明示、用語インライン注)。数値の手入力なし・test 利用示唆なしを監査確認。build/section テスト green
- **WISDM 被験者数削減の再現に着手(Phase 6 / issue #21 DS-1, branch claude/spec-task-planning-ubz1up)**: RQ2 の外的妥当性拡張。subject loader を dispatch 化(`data/subject_datasets.py`: UCI_HAR / WISDM)、WISDM v1.1 ローダ新規(`data/wisdm.py`: 自動DL・不正行スキップ・(subject,activity)区間内200サンプル非重複窓・窓ごとz正規化・seed-0固定split=test12/pool24)。config `wisdm_subject_count.yaml`(target=0.80 を UCI HAR と独立に事前登録、N∈{3,6,9,12,15,18,21,24}×3反復×6拡張=144 runs)。pre_registration/judgment_calls(J-WISDM-*)/preprocessing_notes/reproduction_steps を更新。全 133 テスト green。**グリッド完走(144 runs + negative control 24 runs、失敗0)**。結果: N*(none)=14.0、点推定の最大削減は smote +10.5% だが CI が none と重なり CI 分離で baseline を超える手法なし。label_shuffle 対照は WISDM では削減を示さず(−50.5%)。→ **UCI HAR(F-8/F-10)の結論を第2データセットで再現(F-12)、RQ2 の外的妥当性を補強**。集計を dataset 別に汎用化(reduction.py の phase 依存除去、aggregate に reduction_wisdm 追加)、レポートに §6.4「外的妥当性: WISDM」を自動生成。make all-report / validate green。
- **README 公開実装向け再構成(issue #18, branch claude/spec-task-planning-ubz1up)**: README を「フェーズ実行手順書」から論文公開実装向けへ書き換え。研究の問い(RQ1/RQ2)・主結果(F-5/6/7/8/10/11 の要約表)・再現手順・「やっていないこと/今後の課題」を明記し、進捗/計画記録(Phase 番号・内部 issue 参照・RESEARCH_PLAN リンク)を除去。3ペルソナレビュー反映(SDG 略語廃止→データ拡張、手法グロッサリ・N*/削減率定義・UCR/UCI HAR 正式名追加)。LICENSE・引用節はユーザー判断でいずれも保留。PR #19(draft)。
- **再現・前処理ノートタブ追加(branch claude/spec-task-planning-ubz1up)**: レポートに第3タブ「再現・前処理ノート」を追加。再現手順・データ前処理の補足と不定性・「エイヤッと決めた」判断(sensitivity 付き)・原論文からの実装差分を、`artifacts/{reproduction_steps,preprocessing_notes,judgment_calls}.yaml`・`deviations.md` から自動生成(HTML 手入力なし)。維持手順を skill `/reproducibility-notes` 化し、CLAUDE.md に維持義務を追記。regression テスト追加・全 128 テスト green・ヘッドレス描画で表示確認。
- **PAMAP2 追加に着手(Phase 6 / issue #21 DS-2)**: 横断比較「削減率 vs 母集団被験者数」の少 N 端点(~7-9名)として第3の被験者データ PAMAP2 を追加。設計は `artifacts/ds2_design.md`(test=単一固定 held-out K=3/pool≤6、±16g加速度9ch、100→33.3Hz、~5s(168窓)、protocol 12活動、主指標 macro-F1)。**ステージ1完了**: `data/pamap2.py` 新規(.dat parse/transient除外/protocol12マッピング/9ch選択/NaN窓破棄/stride-3ダウンサンプル/168非重複窓/窓ごとznorm/seed-0 K=3 split/overlapガード、dataset_params 注入、metadata drift 検査 + per-subject 活動サポート記録)、`subject_datasets.py` に PAMAP2 登録、`tests/test_pamap2.py`(18件)。全テスト green。**ステージ2完了(統括 GO 待ちで停止)**: 実データDL(archive sha256=76b3580b…、688MB)+ per-subject 活動サポート実測 → **subject 109 を機械除外**(総窓10・12活動中1のみ=rope_jumping、他 247-301窓/8-12活動、生ファイル3.9MB対118-207MB)。pool=5[101,102,106,107,108]/test=3[103,104,105](seed-0)、subject_counts={2,3,4,5}。config 3種(subject_count 120runs / negative_control 20runs / target_baseline 5runs)+ pre_registration「PAMAP2 DS-2」節(target 選択ルールを数値確定前にコミット済み)。**full-pool none baseline(N=5,5反復)実行**: macro-F1 平均=0.766(sd0.037)、accuracy 平均=0.796(sd0.018)。登録ルール floor_0.05(0.766−0.05)=**target=0.70(macro_f1)**(margin 0.066)。注意: rope_jumping は除外後 pool 4窓・test 0窓で希少(macro-F1 は実効 11 クラス)。**統括 GO 後グリッド完走(DS2-3)**: target=0.70 記入・実験前コミット(b25df99)後、主120+negative control20=**140 runs 完走(失敗0、全 git_dirty=false・単一 commit b25df99)**。集計 `reduction_pamap2`(macro_f1 は pre-registered target 0.70、accuracy は同一ルールで導出 0.70)を results.json に追加。**結果: N*(none) は格子下限 N=2 に左側打ち切り**(none macro-F1 は N=2 で既に平均 0.706≥0.70)→ 測定レンジ {2..5} で**削減率は推定不能**(定性記述)。全6手法とも reduction CI が none と重なり分離せず、label_shuffle は target 未到達で削減を示さず。横断(unified rule): UCI HAR(pool21,N*≈3.4)/WISDM(pool24,N*≈11.9)/PAMAP2(pool5,左側打ち切り)いずれも**削減率は帰無と区別できず**、母集団サイズに沿う系統トレンド非検出=DS-1 帰無の外的妥当性を補強。make test(165)・make validate green。**DS2-4(Reporting レーン)完了**: aggregate に `_cross_dataset_reduction`→results.json['reduction_cross'](3データ×両指標を統一ルール target で再集計、run manifest 駆動・手入力なし)。レポートに §6.5「横断の外的妥当性」を自動生成(横断表8 + 横断図4/5=削減率 vs pool、accuracy/macro_f1 両版、PAMAP2 は左側打ち切りを網掛けで表現し偽の削減値を置かない)。**統一ルールの post-hoc 明示**: 統一 target(macro-F1 で UCI HAR 0.85/WISDM 0.70)は事前登録主 τ(accuracy 0.90/0.80)と異なり、UCI HAR/WISDM では post-hoc の別レンズ・PAMAP2 のみ事前登録どおり、を amber 注記+表8注+F-13+J-CROSS-UNIFIED+limitations に明記。findings に F-13、references に reiss2012、要旨に1文追加。3ペルソナレビュー反映(教授: 表8注に「統一target≠事前登録τ」「UCI HAR 一様正は格子下限圧縮のアーティファクト」を追記/院生: 左側打ち切り・実効11クラスをインライン説明/実務: PAMAP2 DL・109除外根拠は J-PAMAP2-EXCLUDE109 へ導線)。regression 必須 section テストに `subject-reduction-cross` 追加し green。make validate green。
- **WESAD 追加(issue #21 DS-3)完了**: 非HAR 生理信号で信号種依存を検証。§6.6・F-14 反映済み(near-chance=枠組み非転移、削減の有無は語らない)。DS3-1..5 done。
- **外的妥当性アークの統合まとめ(issue #21 §6.7)完了 [本作業]**: 4つの被験者IDデータ(UCI HAR / WISDM / PAMAP2 / WESAD)を1図1表に束ね、削減の外的妥当性を締めた。レポートに §6.7「被験者数削減の外的妥当性:4データ統合」を自動生成——(1) 統合フォレストプロット図7(各データの事前登録 primary target レンズ。測定可能な UCI HAR・WISDM は削減率±CI をプロット、どの手法の CI も none の帯と重複=対照越えなし。PAMAP2=左側打ち切り・WESAD=near-chance は偽の削減値を置かず「推定不能」注記)、(2) 統合表10(データ・pool・信号種・クラス数・主指標・測定可否・結論・対照との関係)。**主結論 F-8/F-10/F-12 は各データ事前登録 τ に基づき不変**を再掲、§6.5 統一ルールは post-hoc 別レンズだが結論一致を明記。記述的・帰無前提に限定(4点=うち2点推定不能で回帰・相関せず、系統トレンド非検出/測定不能、交絡注記)。WESAD は削減率比較でなく「枠組み非転移」の事例として区別。findings に **F-15**(4データ横断の帰無/推定不能・削減できたデータ皆無・DS-5 対象外の根拠)、要旨・まとめに各1文、limitations に測定可能性の境界を追加。synthesis 図表は results.json の各データ reduction ブロックからデータ駆動生成(HTML/数値手入力なし)。regression 必須 section に `subject-reduction-synthesis` 追加し green、make validate green。3ペルソナレビュー反映(下記)。
- **DS-5(マルチモーダル/チャネル横断拡張)は deferred 確定**: 統括判断で「実行せず統合まとめで締める」。理由: 削減を測るには「多モーダル×削減測定可能」な基盤が要るが、多モーダルの PAMAP2 は左側打ち切り・WESAD は near-chance、削減が測れる UCI HAR/WISDM は単一モダリティ寄りで、該当データが無い。将来の発火条件(多モーダルで N*(none) 格子内推定可能かつ label_shuffle≪none)を issue #27 に残す。評価は `artifacts/ds5_assessment.md`。DS-4 は本アークの範囲外。§6.7 の対象外根拠ボックス + F-15 notes + limitations に明記。
- 次の候補: Phase 6-7(削減評価の反復数増による確度向上、拡張強度スイープ、統合レポート、WESAD 生理向け前処理 ablation=別 issue)。詳細は GitHub issue #7 / #27

## フェーズ別の経過(時系列)

### Phase 0 完了
基盤構築(config 駆動・runner・manifest・監査・Tailwind HTML 自動生成)。品質ゲート(smoke)通過。

### Phase 1 完了
126 runs(3データセット × 7拡張 × 2モデル × 3 seeds)全完走・監査 130/130 合格。知見 F-1〜F-4: フルサイズ学習データでは改善は +1pt 未満・悪化は最大 -3.5pt、GunPoint でのみ改善傾向(ノイズ範囲)、FordA+MiniRocket では拡張が有害。ベースラインは文献相場と整合。3ペルソナレビューを反映。

### Phase 2 完了
12データセット × 5学習比率 × 7拡張 × 2モデル × 3 seeds = 2520 runs 全完走・監査合格。知見 F-5(1D-CNN では拡張が Holm 補正後も有意)、F-6(MiniRocket ではどの手法も非有意)、F-7(効果は低〜中比率で相対的に大)。
- 初回実行は runner の二重起動で同一 run_id を並行書き込みし競合クラッシュ(約183 runs 時点)。対策: pid ロック(runs/.runner.lock)で二重起動拒否 + manifest の tmp 名を pid 付きに変更(test_grid_lock.py で担保)。単一プロセスで resume し完走。

### Phase 3 完了
被験者ID付き公開データを調査(全 CC BY 4.0)。主対象 UCI HAR(30名)、第2の被験者データ WISDM v1.1(Kwapisz et al. 2011, 36名)を決定(より大規模な 2019 年 biometrics 版 51名とは別データ)。

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
