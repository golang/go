// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (arm64 || riscv64)

package unix

// This file is named "generic" because at a certain point Linux started
// standardizing on system call numbers across architectures. So far this
// means only arm64 and riscv64 use the standard numbers.

const (
	getrandomTrap     uintptr = 278
	copyFileRangeTrap uintptr = 285
)
