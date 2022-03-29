// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && !386 && !arm

package syscall

const (
	sys_GETEUID = SYS_GETEUID

	sys_SETGID = SYS_SETGID
	sys_SETUID = SYS_SETUID

	sys_SETREGID = SYS_SETREGID
	sys_SETREUID = SYS_SETREUID

	sys_SETRESGID = SYS_SETRESGID
	sys_SETRESUID = SYS_SETRESUID
)
