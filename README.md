# Contribution Guidelines for Go RISC-V Repository
**English Version** | [中文版本](README-CN.md)

Repository: https://github.com/zte-riscv/go

## 1. Introduction
This specification applies to the Go RISC-V code repository jointly maintained by ZTE and ByteDance. We are committed to enhancing Golang support for RISC-V architecture, regularly contributing code upstream to [golang/go](https://github.com/golang/go) while synchronizing with its latest features.

We warmly welcome community developers to participate! Whether it's fixing bugs, implementing new features, or improving documentation, your contributions will help advance the RISC-V ecosystem.

To ensure efficient collaboration, please review the following guidelines before submitting code:

## 2. Version Download and Installation

### 2.1 Download Version
Download the corresponding version from the [GitHub Releases page](https://github.com/zte-riscv/go/releases), selecting the version suitable for your system architecture (e.g., `linux-riscv64.tar.gz`).

The naming rules for versions are：go version number+version release date "YYYYMM", separated by '.', for example: 1.25.3.202512. The version release cycle is one month.

### 2.2 Extract and Install
Extract the downloaded archive to the target directory:
```bash
# Extract to a specified directory (e.g., /usr/local)
tar -C /usr/local -xzf go1.xx.x.linux-riscv64.tar.gz
```

### 2.3 Configure GOROOT
Configure the `GOROOT` environment variable to point to the Go installation directory:

**Method 1: Temporary Configuration (valid for current session)**
```bash
export GOROOT=/usr/local/go
export PATH=$PATH:$GOROOT/bin
```

**Method 2: Permanent Configuration (Recommended)**
Add the following to `~/.bashrc` or `~/.bash_profile`:
```bash
export GOROOT=/usr/local/go
export PATH=$PATH:$GOROOT/bin
```
Then execute `source ~/.bashrc` to apply the configuration.

**Method 3: System-wide Configuration**
Edit `/etc/profile` or `/etc/environment` and add the above environment variable configuration.

### 2.4 Verify Installation
After installation, verify with the following command:
```bash
go version
```
If configured correctly, it will display output similar to `go version go1.xx.x linux/riscv64`.

## 3. Pre-Submission Preparation
### 3.1 Issue Tracking and Discussion
**Mandatory**:
- Create or claim a related Issue before commencing substantive work
- Clearly describe the change motivation, design approach, and expected impact in the Issue

**Recommended**:
- For significant changes (e.g., new instruction set support, architectural modifications), submit a design document for prior review

### 3.2 Branch Management
```bash
# Create a feature branch from the current development branch (e.g., go1.26.3-zte-dev):
git fetch origin
git checkout -b your_dev_branch origin/go1.26.3-zte-dev
```

## 4. Code Contribution Process
### 4.1 Development Workflow
**Issue Submission**:
- For bug reports: Document the problem, environment, and reproduction steps
- For features/optimizations: Describe the technical background and implementation strategy

**Pull Request Submission**:
- Provide task context in the PR description and reference related Issues
- Resolve all merge conflicts locally before submission

**CI and Review**:
- Automated test suites will execute upon PR submission (failed tests block merging)
- Require approvals from designated reviewers before merging, reviewers comment +1 or +2.

**Merge Policy**:
- All commits must be squashed into a single commit

### 4.2 Commit Message Standards
For squashed commits, follow this format:
```bash
git commit -m "package: concise change summary

Fixes #12345
Updates #67890"
```
- **Header**: Package prefix + brief description (<50 chars)
- **Footer**: Issue references

## 5. Review Process
### 5.1 Mandatory Requirements
- CI pipeline must pass
- Minimum approvals:
- One +2 from ZTE core maintainers
- One +2 from ByteDance core maintainers

### 5.2 Review Tiers
- `+1`: Preliminary approval (requires additional reviewer)
- `+2`: Final approval for merging

**For maintainers and reviewers: 
when approving the merge, please respond with either +1 or +2.**

### 5.3 Core Maintainers
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

### 5.4 Review Checklist
- ✅ Go coding standards compliance
- ✅ Passing CI pipelines
- ✅ Corresponding test coverage
- ✅ Performance benchmarks (where applicable)

## 6. CI Pipeline Requirements
### 6.1 Test
- **Static Analysis**: Run `go vet` on all modified code packages
- **Codegen Validation**: Verify RISCV64 instruction generation via `asmcheck` (`go/test/codegen/`)
- **Assembly Tests**: Validate machine code output from assembly instructions (`go/src/cmd/asm/internal/asm/testdata/riscv64.s`)
- **Regression Testing**: Execute `go/src/all.bash` on x86/ARM architectures
- **Application Testing**: Validate test file execution in QEMU-RISCV environment (`go/test/zte/*_test.go`)

## 7. Performance Reporting (Optional)
### 7.1 Benchmark Requirements
For performance-sensitive changes:

Test Environment:
```markdown
- Hardware: [e.g., SiFive Unmatched, VisionFive2]
- Go Version:
- Test Date:
- CPU (Optional): [Cores, frequency, RISC-V extensions]
- Memory (Optional):
```
Benchmark Results, `benchstat` comparison:
```bash
benchstat old.txt new.txt

name        old time/op     new time/op    delta    
Fibonacci   125ms ± 2%      98ms ± 3%      -21.60% (p=0.000 n=10+10) 
Sort        456ms ± 1%      401ms± 2%      -12.06% (p=0.000 n=9+10)

```
## 8. Appendix
Official Go Contribution Guide: https://golang.org/doc/contribute
