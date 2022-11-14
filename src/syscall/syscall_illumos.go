// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build illumos

package syscall

import "unsafe"

// F_DUP2FD_CLOEXEC has different values on Solaris and Illumos.
const F_DUP2FD_CLOEXEC = 0x24

//go:cgo_import_dynamic libc_flock flock "libc.so"

//go:linkname procFlock libc_flock

var procFlock libcFunc

func Flock(fd int, how int) error {
	_, _, errno := sysvicall6(uintptr(unsafe.Pointer(&procFlock)), 2, uintptr(fd), uintptr(how), 0, 0, 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}
