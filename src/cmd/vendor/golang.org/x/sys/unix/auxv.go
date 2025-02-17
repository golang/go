// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.21 && (aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || zos)

package unix

import (
	"syscall"
	"unsafe"
)

//go:linkname runtime_getAuxv runtime.getAuxv
func runtime_getAuxv() []uintptr

// Auxv returns the ELF auxiliary vector as a sequence of key/value pairs.
// The returned slice is always a fresh copy, owned by the caller.
// It returns an error on non-ELF platforms, or if the auxiliary vector cannot be accessed,
// which happens in some locked-down environments and build modes.
func Auxv() ([][2]uintptr, error) {
	vec := runtime_getAuxv()
	vecLen := len(vec)

	if vecLen == 0 {
		return nil, syscall.ENOENT
	}

	if vecLen%2 != 0 {
		return nil, syscall.EINVAL
	}

	result := make([]uintptr, vecLen)
	copy(result, vec)
	return unsafe.Slice((*[2]uintptr)(unsafe.Pointer(&result[0])), vecLen/2), nil
}
