// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && !(loong64 || riscv64)

package unix

import "syscall"

const (
	renameatTrap uintptr = syscall.SYS_RENAMEAT
)
