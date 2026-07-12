# state.md

最終更新: 2026-07-12(Phase 1 完了)

## 現在の状態

- Phase 0 完了(品質ゲート通過、テスト 51 件全通過)
- **Phase 1 完了**: 126 runs(3データセット × 7拡張 × 2モデル × 3 seeds)全完走・失敗 0・監査 130/130 合格
- 主要知見は artifacts/findings.json(F-1〜F-4)。フルサイズ学習データでは拡張効果は僅少、最小データセット GunPoint でのみ改善、FordA+MiniRocket では拡張が有害
- 次: レポートのペルソナレビュー(P1-7)→ Phase 2 準備(P2-1: データセット選定)

## 環境

- Python 3.11 / uv 管理(torch CPU 版・aeon 1.5)
- 実行環境: Claude Code リモートコンテナ(CPU のみ)
- Phase 1 の実測実行時間: 約 1.5 時間(FordA の CNN が支配的)

## 注意事項

- report/dist/index.html は Git 管理する方針(ユーザー決定 2026-07-12)
- LICENSE の選定は保留中(ユーザー決定 2026-07-12)
- Phase 2 の学習サンプル比率スイープには runner の拡張(train_fractions 対応)が必要
