// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build illumos

// Illumos system calls not present on Solaris.

package syscall

import "unsafe"

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
