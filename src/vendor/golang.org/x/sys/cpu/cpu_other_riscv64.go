// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux && riscv64
// +build !linux,riscv64

package cpu

func archInit() {
	Initialized = true
}
