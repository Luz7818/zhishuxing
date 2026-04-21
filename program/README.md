<div align="center">
<p align="center">
  <img width="100%" height="350" src="https://raw.gitcode.com/qq_51399582/MARL/attachment/uploads/9666103a-af99-4c09-b1f1-8f19a07bfcea/image.png" />
</p>

# 多智能体强化学习算法库
## Multi-Agent Deep Reinforcement Learning，MADRL

**算法库包含多个用于多智能体强化学习（MARL）的算法实现，如 **IQL、VDN、QMIX、MADDPG、MATD3、MAPPO**等**

---

[English](README.md) | [简体中文](README.md)

<!-- prettier-ignore -->
<a href="https://gitcode.com/qq_51399582/MARL/blob/main/README.md">文档</a> •
<a href="https://gitcode.com/qq_51399582/MARL/tree/main/1.IQL">示例</a> •
<a href="https://blog.csdn.net/qq_51399582?spm=1000.2115.3001.5343">博客</a> •
<a href="https://bbs.csdn.net/forums/RL?spm=1001.2014.3001.6685">社区</a>


[![star](https://gitcode.com/qq_51399582/MARL/star/badge.svg)](https://github.com/your-repository)
[![Downloads](https://static.pepy.tech/badge/your-package)](200)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://bbs.csdn.net/forums/RL?spm=1001.2014.3001.6685)
[![Twitter](https://img.shields.io/twitter/follow/your-twitter?style=social)](https://wx.mail.qq.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

## 项目概述
多智能体系统通常需要多个智能体在一个共享环境中进行协作与竞争，相较于单智能体强化学习，MARL问题更为复杂。本库主要实现了一些经典的MARL算法，旨在帮助研究人员和开发者在多智能体环境中实现和测试强化学习算法。

#### [【MADRL】多智能体深度强化学习《纲要》](https://rainbook.blog.csdn.net/article/details/141318848)

## 项目背景
多智能体深度强化学习将深度学习与多智能体强化学习结合，使得智能体能够在复杂、高维的环境中学习到有效的策略。MADRL 涉及多个智能体在共享环境中进行交互，这些智能体可能具有不同的目标、信息和能力，因此相较于单智能体强化学习问题，MADRL 更加复杂且具有挑战性。


## 算法简介

以下是本算法库中实现的主要算法，每种算法都包括了详细的理论介绍和代码实现：

| **算法名称**                               | **描述**  | **链接**    |
|--------------------------------------------|------------------------------------|------------------------------------------------------------------|
| **独立 Q 学习 (IQL)**                      | 通过独立地训练每个智能体的 Q 网络来进行学习| [阅读文章](https://rainbook.blog.csdn.net/article/details/141380210) |
| **基于 MADRL 的单调价值函数分解 (QMIX)**    | 通过分解全局价值函数来实现多智能体协作的算法。| [阅读文章](https://rainbook.blog.csdn.net/article/details/141395534) |
| **多智能体深度确定性策略梯度 (MADDPG)**     | 采用深度确定性策略梯度 (DDPG) 作为基础，扩展到多智能体场景的算法。| [阅读文章](https://rainbook.blog.csdn.net/article/details/141996518) |
| **多智能体双延迟深度确定性策略梯度 (MATD3)** | 在 MADDPG 的基础上，使用双延迟机制提升训练稳定性。| [阅读文章](https://rainbook.blog.csdn.net/article/details/141997347) |
| **多智能体近似策略优化 (MAPPO)**           | 基于策略梯度的方法，采用近似策略优化技术来改进多智能体学习效率。| [阅读文章](https://rainbook.blog.csdn.net/article/details/142068153) |
| **反事实多智能体策略梯度 (COMA)**          | 使用反事实概念来调整策略，使得每个智能体都能考虑到其他智能体的行动。| [阅读文章](https://rainbook.blog.csdn.net/article/details/142109244) |
| **多智能体价值分解网络 (VDN)**             | 通过将全局 Q 值分解成局部 Q 值来简化学习过程。| [阅读文章](https://rainbook.blog.csdn.net/article/details/142146917) |
| **多智能体信任域策略优化 (MA-TRPO)**       | 在 TRPO 的基础上引入了信任域机制，提升多智能体系统的稳定性。| [阅读文章](https://rainbook.blog.csdn.net/article/details/142304128) |
| **面向角色的多智能体强化学习 (ROMA)**      | 基于角色定义的策略优化，能够有效处理多智能体环境中的复杂互动。| [阅读文章](https://rainbook.blog.csdn.net/article/details/142338261) |

## 项目配置

在本项目中，您将需要以下依赖包来运行相关算法代码：

- Python 版本： `3.11.5`
- PyTorch 版本： `2.1.0`
- torchvision 版本： `0.16.0`
- gym 版本： `0.26.2`
- tensorboardX

您可以使用以下命令来安装这些依赖：
```bash
pip install -r requirements.txt
```

## 综合交通枢纽快速换乘（ML-Agents + MADDPG）

本仓库中的 `MADDPG` 已接入 `mlagents-envs`，可直接用于 Unity 侧构建的综合交通枢纽快速换乘场景（多智能体协同调度、引导、疏散、换乘优化等）。

### 1) Unity 场景侧约束

- 使用 **连续动作空间**（MADDPG 当前实现仅支持连续动作）。
- 所有协同智能体挂载同一个 `Behavior Name`。
- 每个 Agent 提供可拼接的一维或多维观测（代码会自动展平拼接）。
- Reward 建议包含：换乘时间惩罚、拥堵惩罚、协同效率奖励、冲突惩罚。

### 2) 训练命令（Python 侧）

在 `program` 根目录执行：

```bash
python .\MADDPG\MADDPG_main.py --mlagents_file "D:\\Builds\\HubTransfer\\HubTransfer.exe" --behavior_name "HubAgent" --base_port 5005 --episode_limit 200 --max_train_steps 500000 --evaluate_freq 5000
```

若连接 Unity Editor（而非 exe），可将 `--mlagents_file` 留空（默认 `None`），并确保 Editor 正在 Play。

### 3) 输出结果

- 训练曲线：`runs/`
- 评估奖励：`data_train/`
- 模型权重：`model/<env_name>/`

## 交流方式

如果对该算法有何问题，可通过以下方式交流：

- 📢 CSDN博客主页私信：[不去幼儿园](https://blog.csdn.net/qq_51399582?type=blog)  
- 📘 CSDN强化学习社区：[强化学习交流社区](https://bbs.csdn.net/forums/RL?spm=1001.2014.3001.6685) 
- 💻 微信：Rainbook_2