// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// On 32-bit Linux systems, use SYS_FCNTL64.
// If you change the build tags here, see syscall/flock_linux_32bit.go.

//go:build (linux && 386) || (linux && arm) || (linux && mips) || (linux && mipsle)

package unix

import "syscall"

func init() {
	FcntlSyscall = syscall.SYS_FCNTL64
}
