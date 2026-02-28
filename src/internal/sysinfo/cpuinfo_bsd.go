// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || freebsd || netbsd || openbsd

package sysinfo

import "syscall"

func osCPUInfoName() string {
	cpu, _ := syscall.Sysctl("machdep.cpu.brand_string")
	return cpu
}
