# Hybrid Video Retargeting

コンテンツ認識型ビデオリサイズ処理のCPU/CUDA実装

## 概要

このプロジェクトは、画像の意味的な重要性を考慮しながらビデオのサイズを変更する手法を実装したものです。以下の3つの主要な処理を組み合わせることで、高品質なビデオリサイズを実現します：

1. **セグメンテーション**: SLICアルゴリズムを用いて画像をスーパーピクセルに分割
2. **顕著性マップ計算**: 色、位置、エッジ特徴を考慮した顕著性マップの生成
3. **コンテンツ適応型ワーピング**: エネルギー最小化に基づく画像変形

特徴：

- CPU/CUDA実装の両方をサポート
- OpenMPを使用したCPU並列処理
- 最新のGPUアーキテクチャ（Turing以降）に対応
- 高速な処理と高品質な結果の両立

## 必要条件

### CPU版

- C++17対応コンパイラ
- OpenCV 4.0以降
- OpenMP対応コンパイラ (推奨)
- CMake 3.18以降

### CUDA版 (上記CPU版の条件に加えて)

- CUDA Toolkit 11.0以降
- OpenCV 4.0以降 (**CUDA対応ビルドが必須**)

## ビルド手順

```bash
# プロジェクトのクローン
git clone https://github.com/yourusername/video_retargeting_cuda.git
cd video_retargeting_cuda

# ビルドディレクトリの作成
mkdir build
cd build

# CMakeの実行
cmake ..

# ビルド
make -j$(nproc)

# インストール（オプション）
sudo make install
```

## 使用方法

基本的な使用方法：

```bash
./video_retargeting <入力ビデオ> <出力ビデオ> <出力幅> <出力高さ> [--cuda]
```

例：

```bash
# CPU版で実行（1280x720にリサイズ）
./video_retargeting input.mp4 output.mp4 1280 720

# CUDA版で実行（1920x1080にリサイズ）
./video_retargeting input.mp4 output.mp4 1920 1080 --cuda
```

## ライセンス

MIT License

## 謝辞

このプロジェクトは以下の論文の実装を参考にしています：
- Lin, S. S., Lin, C. H., Yeh, I. C., Chang, S. H., Yeh, C. K., & Lee, T. Y. (2013). Content-Aware Video Retargeting Using Object-Preserving Warping. *IEEE Transactions on Visualization and Computer Graphics, 19*(12), 2196-2205.
- Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., & Süsstrunk, S. (2012). SLIC Superpixels Compared to State-of-the-Art Superpixel Methods. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 34*(11), 2274-2282.
