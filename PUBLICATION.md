# 本地推理生产线②：双卡 TP2 主力线

- **角色**：双卡（2×RTX 5060 Ti）张量并行跑 27B，主力对话/工具线。
- **模型工件（未包含在本仓）**：`qwen3_8_27b_coder390_w4a4w8a8.ninfer`，21.5 GB（Qwen3.8-27B Coder390 W4A4-W8A8）。
- **生产配置**：端口 `8091`，ctx `262144`，MTP 投机 K=3，前缀复用 ≈99.9%，单元名 `ninfer-tp2`（**由路由器按需拉起的瞬时单元**，`systemd-run --user --collect`，仓内无静态 unit 文件）。
- **本仓内容**：引擎源码快照（**不含**编译产物；生产二进制 209 MB 不进仓）。含 **XGrammar v0.2.7** vendored（受限解码）与本线新增的结构化输出实现。
- **来源与许可**：fork 自 `BaiYouQing/ninfer-tp2-5060ti-windows`（公开，**Apache-2.0**，见 `LICENSE` / `NOTICE`）。
- **快照点**：分支 `windows-port` @ `8c50e4ed`（2026-10-13），本仓库以 `main` 发布。
