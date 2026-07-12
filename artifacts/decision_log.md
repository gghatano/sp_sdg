# decision_log.md

意思決定の記録。形式: 日付 / 決定 / 理由 / 影響範囲

- 2026-07-12 / 計画ドキュメントを docs/plan.md に配置 / spec §4 の RESEARCH_PLAN.md(研究設計)と役割分離するため / ドキュメント構成
- 2026-07-12 / report/dist/index.html を Git 管理する / ユーザー決定(spec §4.3 の初期方針どおり) / .gitignore
- 2026-07-12 / LICENSE 選定は後回し / ユーザー決定 / ルートファイル
- 2026-07-12 / mixup は同一クラスペアに限定(ソフトラベル不使用) / RidgeClassifier がソフトラベル非対応のため実装を統一。原論文との差分は deviations.md に記録 / augmentations
- 2026-07-12 / DTW 拡張は同一クラスペアの DTW 整列平均で実装 / Forestier et al. 2017 の加重平均法の簡略版。計算コストと効果のバランス / augmentations
- 2026-07-12 / smoke 用 CNN は 15 epochs / 3 epochs では合成データでも未学習(精度 0.475)だったため / config/models.yaml
- 2026-07-12 / 検証データは train からの層化分割のみ(val_fraction) / test への early stopping リーケージ防止(spec §8) / models
