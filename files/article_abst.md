# Least Squares Generative Adversarial Networks

- [Arxiv](https://arxiv.org/pdf/1611.04076.pdf)
- GANに関する論文
- 論文実装の前に一旦概要をまとめておく


## 概要

- Least Squares GANは正解ラベルに対する二乗誤差を用いる学習手法の提案


## Abstract

- GANには通常誤差関数としてシグモイドクロスエントロピー誤差が用いられるが，学習過程で勾配消失が起こりうる．
- それを解決すべく，最小二乗を用いたGANの提案を行う．
- LSGANの目的関数を最小化すると、ピアソンのχ2発散が最小化されることを示す．
- LSGANが正規化GANよりも優れている点は
	- 生成される画像のクオリティが高い
	- 学習過程で安定したパフォーマンスを得られる

## Introduction

- GANはend-to-endで学習することができる．
- GANの基本的なアイデアは，分類器と生成器を同時に学習させる: 生成器がfake sampleを生成している間，分類器はreal sampleとgenerated sampleの判別をする; 生成器は分類器に本物のデータとフェイクデータの判別が難しくなるように生成する

![](./img/lsgan.png)

- 
