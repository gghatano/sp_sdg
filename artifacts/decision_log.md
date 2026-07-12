# decision_log.md

意思決定の記録。形式: 日付 / 決定 / 理由 / 影響範囲

- 2026-07-12 / 計画ドキュメントを docs/plan.md に配置 / spec §4 の RESEARCH_PLAN.md(研究設計)と役割分離するため / ドキュメント構成
- 2026-07-12 / report/dist/index.html を Git 管理する / ユーザー決定(spec §4.3 の初期方針どおり) / .gitignore
- 2026-07-12 / LICENSE 選定は後回し / ユーザー決定 / ルートファイル
- 2026-07-12 / mixup は同一クラスペアに限定(ソフトラベル不使用) / RidgeClassifier がソフトラベル非対応のため実装を統一。原論文との差分は deviations.md に記録 / augmentations
- 2026-07-12 / DTW 拡張は同一クラスペアの DTW 整列平均で実装 / Forestier et al. 2017 の加重平均法の簡略版。計算コストと効果のバランス / augmentations
- 2026-07-12 / smoke 用 CNN は 15 epochs / 3 epochs では合成データでも未学習(精度 0.475)だったため / config/models.yaml
- 2026-07-12 / 検証データは train からの層化分割のみ(val_fraction) / test への early stopping リーケージ防止(spec §8) / models
- 2026-07-12 / Phase 2 データセット 12 件を選定(GunPoint, ECG5000, ECG200, TwoLeadECG, ItalyPowerDemand, MoteStrain, SonyAIBORobotSurface1, CBF, Plane, ArrowHead, Coffee, Wafer) / 選定基準: (1) 学習データが小〜中規模で低データ条件を検証しやすい (2) ドメイン多様性(ECG・センサ・動作・シミュレーション・分光・半導体) (3) 2値と多クラスの混在 (4) Phase 1 との連続性(GunPoint, ECG5000 継続) (5) CPU 実行時間(FordA は学習コスト過大のため除外) / config/experiments/phase2.yaml
- 2026-07-12 / run_id の学習比率タグは 1.0 のとき省略 / Phase 1 の既存 run_id と resume 互換性を保つため / runner
- 2026-07-12 / 統計検定は Wilcoxon 符号順位検定(対応: dataset×model×seed×fraction、5 ペア未満はスキップ) / RESEARCH_PLAN.md の設計どおり。多重比較補正は Phase 2 集計時に検討 / evaluation/stats.py
