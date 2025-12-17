# AimDK 接口协议

## 简介

本仓库定义了 AgiBot 产品的所有跨模块的接口协议，包含对外的部分与仅在内部流通的部分。

本协议的根命名空间是  **aimdk::protocol::** ，我们按不同模块进一步划分了子目录，并以这些模块的名，命名这些子目录及其下一级命名空间。



## 约定

协议编写规范见 [《研发规划/协议规范》](https://agirobot.feishu.cn/wiki/YBe5wYMtiicci2kBgBxc8KUHngb?fromScene=spaceOverview&theme=LIGHT&contentTheme=DARK) 。

我们需要避免冗余协议内容的出现，若多个模块出现相同的协议内容，若其语义拥有公共属性，应将其放到 common 子协议目录下。

我们将每个协议内容的定义，以注释的形式放在协议源文件中，但在每个子目录（包括本目录）设立 README.md，作为简单的检索之用，
让阅读者更方便找到所需的协议文件。



## 模块目录

- [公共部分](./common/README.md)
- [定位模块（SLAM）](./slam/README.md)
