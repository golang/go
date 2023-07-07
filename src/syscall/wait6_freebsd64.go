// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd && (amd64 || arm64 || riscv64)

package syscall

import (
	"unsafe"
)

const _P_PID = 0

func wait6(idtype, id, options int) (status int, errno Errno) {
	var status32 int32 // C.int
	_, _, errno = Syscall6(SYS_WAIT6, uintptr(idtype), uintptr(id), uintptr(unsafe.Pointer(&status32)), uintptr(options), 0, 0)
	return int(status32), errno
}
