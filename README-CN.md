# Go RISC-V代码仓库贡献指南
[English Version](README.md) | **中文版本**

代码仓库：https://github.com/zte-riscv/go

## 1. 引言
本规范适用于中兴与字节跳动共同维护的 Go RISC-V 代码仓库。我们致力于完善 Golang 在 RISC-V 架构的支持，并定期将代码贡献回上游 [golang/go](https://github.com/golang/go) 仓库，同时同步其最新特性。

我们非常欢迎社区开发者参与项目！无论是修复 Bug、实现新功能，还是改进文档，您的贡献都将帮助推动 RISC-V 生态的发展。

为确保协作高效顺畅，请在提交代码前阅读以下指南：


## 2. 提交前准备
### 2.1 问题跟踪与讨论
**必须**：
- 在开始实质性工作前创建或认领相关Issue
- 在Issue中明确描述变更动机、设计方案和预期影响

**建议**：
- 对于重大变更（如新指令集支持、架构修改），提前提交设计文档进行评审

### 2.2 分支管理
```bash
# 从当前开发分支（如go1.25.0-zte-dev）创建特性分支：
git fetch origin
git checkout -b your_dev_branch origin/go1.25.0-zte-dev
```

## 3. 代码贡献流程
### 3.1 开发工作流
**Issue提交**：
- 错误报告：记录问题现象、环境及复现步骤
- 功能/优化：说明技术背景和实现策略

**Pull Request提交**：
- 在PR描述中提供任务背景并关联相关Issue
- 提交前在本地解决所有合并冲突

**CI与评审**：
- PR提交后将自动执行测试套件（测试失败将阻止合并）
- 需获得指定评审人的批准方可合并，评审人回复+1或+2。

**合并策略**：
- 所有提交以sqash merge的方式压缩为单个commit后合入目标分支

### 3.2 提交信息规范
压缩提交时请遵循以下格式：
```bash
git commit -m "包名: 简洁的变更摘要

修复 #12345
更新 #67890"
```
- **标题**：包名前缀 + 简要说明（<50字符）
- **页脚**：包含相关issue引用

## 4. 评审流程
### 4.1 强制要求
- CI流水线必须通过
- 最低审批要求：
- 1名中兴核心维护者的+2
- 1名字节跳动核心维护者的+2

### 4.2 评审等级
- `+1`：初步批准（需额外评审人）
- `+2`：最终批准（可合并）

**对维护者和评审人的要求：
批准合并时，请回复+1或+2。**

### 4.3 核心维护团队
| GitHub | Role | Org | Auth Level |
|--------|------|-----|-----------|
| @agiledragon | Maintainer | ZTE | +2 |
| @ctk-1998 | Reviewer | ZTE | +1 |
| @lxq015 | Reviewer | ZTE | +1 |
| @hehongjun20110618 | Maintainer | ZTE | +2 |
| @newborn22 | Maintainer | ZTE | +2 |
| @wenchangping | Reviewer | ZTE | +1 |
| @wangpc-pp | Maintainer | ByteDance | +2 |
| @BoyaoWang430 | Reviewer | ByteDance | +1 |

### 4.4 评审检查清单
- ✅ 符合Go编码规范
- ✅ 通过CI流水线
- ✅ 包含对应测试用例
- ✅ 性能基准测试（如适用）

## 5. CI流水线要求
### 5.1 测试要求
- **静态分析**：对涉及修改的代码包执行`go vet`
- **代码生成验证**：通过`asmcheck`验证RISCV64架构下是否生成了正确的指令（`go/test/codegen/`）
- **汇编测试**：验证汇编指令是否生成正确的机器码（`go/src/cmd/asm/internal/asm/testdata/riscv64.s`）
- **回归测试**：在x86/ARM架构执行`go/src/all.bash`
- **应用测试**：QEMU-RISCV环境下验证测试文件是否正确执行（`go/test/zte/*_test.go`）

## 6. 性能报告（可选）
### 6.1 基准测试要求
对性能敏感变更需提供如下信息：

测试环境：
```markdown
- 硬件：[如SiFive Unmatched, VisionFive2]
- Go版本：
- 测试日期：
- CPU（可选）：[核心数、频率、RISC-V扩展指令]
- 内存（可选）：
```

基准测试结果，`benchstat`对比：
```bash
benchstat old.txt new.txt

name        old time/op     new time/op    delta    
Fibonacci   125ms ± 2%      98ms ± 3%      -21.60% (p=0.000 n=10+10) 
Sort        456ms ± 1%      401ms± 2%      -12.06% (p=0.000 n=9+10)

```

## 7. 附录
官方Go贡献指南：https://golang.org/doc/contribute

