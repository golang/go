// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build 386 arm

package syscall

const (
	sys_GETEUID = SYS_GETEUID32

	sys_SETGID = SYS_SETGID32
	sys_SETUID = SYS_SETUID32

	sys_SETREGID = SYS_SETREGID32
	sys_SETREUID = SYS_SETREUID32

	sys_SETRESGID = SYS_SETRESGID32
	sys_SETRESUID = SYS_SETRESUID32
)
