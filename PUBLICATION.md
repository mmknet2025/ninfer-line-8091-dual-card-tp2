# 本地推理生产线②：双卡 TP2 主力线

> 最后同步：**2026-10-29 14:40**

## 角色

双卡（2×RTX 5060 Ti 16 GB，Gen5 x8+x8，P2P 可用）张量并行跑 27B，主力对话/工具线。

## 生产配置（现役）

- 端口 `8091`，`--tp 2 --devices 0,1`，ctx `262144`，`--kv-dtype int8`
- 投机 `--spec mtp --draft-tokens 3 --lm-head-draft`
- `--structured-output`、`--max-concurrency 1`、`--prefill-chunk 1024`
- 单元 `ninfer-tp2`（**由路由器按需拉起的瞬时单元**，`systemd-run --user --collect`，仓内无静态 unit 文件）
- 冷启 ≈8.5 s

## 模型工件（不进本仓）

`qwen3_8_27b_coder390_w4a4w8a8.ninfer`，21,490,955,393 B（Qwen3.8-27B Coder390 W4A4-W8A8，NVFP4）。
sha256 `e15f02938b69…` 与 HF 上游 lfs 一致。

## 本仓内容

引擎源码快照（**不含**编译产物；生产二进制 216,109,952 B 不进仓）。含 **XGrammar v0.2.7** vendored（受限解码）+ third_party 82 个文件 + 本线新增的 **structured-output 实现**（`src/serve/structured_output.{cpp,h}`、`src/text/structured_output.{cpp,h}` 等 37 个文件，上游没有）。

⚠️ **不要在本仓直接切到上游分支** —— 上游没有这些移植与 vendored 依赖，切过去会丢 4.4 万行。

## 版本状态

| 项 | 值 |
|---|---|
| 本机源码 | 分支 `so-port-structured-output-recovered` @ **`74a2879b`**（2026-10-21） |
| 生产二进制 | **`e630612d`**（216,109,952 B，2026-10-29 01:09 重编，CUDA 13.4.92 / `120a` / Ninja Release） |
| 上游 | `BaiYouQing/ninfer-tp2-5060ti-windows` @ `c50ccc91`（**2026-09-23，已停更**） |
| 与上游关系 | 本地领先 1 个提交（移植），落后 0 ⇒ **无上游新内容可跟** |

## 参考性能（冷前缀，greedy，3 rep 中位）

| 场景 | prefill | decode | TTFT |
|---|---:|---:|---:|
| 短 prompt（~740 tok） | 2,862 tok/s | 79.1 tok/s | 264 ms |
| 长 prompt（~6.2k tok） | 3,564 tok/s | 74.4 tok/s | 1,759 ms |

（同一工件在官方 TP 引擎上只有 1,103 / 72.1 / 670 ms 与 1,283 / 77.5 / 4,502 ms —— 本线的专用 TP 内核在 prefill 上快约 2.6 倍。）

## 来源与许可

fork 自 `BaiYouQing/ninfer-tp2-5060ti-windows`（公开，**Apache-2.0**，见 `LICENSE` / `NOTICE`）。

## 快照点

- 本机：`74a2879b`（2026-10-21）
- 上版：分支 `windows-port` @ `8c50e4ed`（2026-10-13）

## 不含什么

编译产物与模型权重都不进仓。
