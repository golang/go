// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && !ios

package unix

import (
	"internal/abi"
	"unsafe"
)

//go:cgo_import_dynamic libc_getentropy getentropy "/usr/lib/libSystem.B.dylib"

func libc_getentropy_trampoline()

// GetEntropy calls the macOS getentropy system call.
func GetEntropy(p []byte) error {
	_, _, errno := syscall_syscall(abi.FuncPCABI0(libc_getentropy_trampoline),
		uintptr(unsafe.Pointer(&p[0])),
		uintptr(len(p)),
		0)
	if errno != 0 {
		return errno
	}
	return nil
}
