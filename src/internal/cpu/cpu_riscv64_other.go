// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64 && !linux

package cpu

func osInit() {
	// Other operating systems do not support the riscv_hwprobe syscall.
}
