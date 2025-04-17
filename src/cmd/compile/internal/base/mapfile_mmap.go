// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package base

import (
	"internal/unsafeheader"
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

// TODO(mdempsky): Is there a higher-level abstraction that still
// works well for iimport?

// MapFile returns length bytes from the file starting at the
// specified offset as a string.
func MapFile(f *os.File, offset, length int64) (string, error) {
	// POSIX mmap: "The implementation may require that off is a
	// multiple of the page size."
	x := offset & int64(os.Getpagesize()-1)
	offset -= x
	length += x

	buf, err := syscall.Mmap(int(f.Fd()), offset, int(length), syscall.PROT_READ, syscall.MAP_SHARED)
	runtime.KeepAlive(f)
	if err != nil {
		return "", err
	}

	buf = buf[x:]
	pSlice := (*unsafeheader.Slice)(unsafe.Pointer(&buf))

	var res string
	pString := (*unsafeheader.String)(unsafe.Pointer(&res))

	pString.Data = pSlice.Data
	pString.Len = pSlice.Len

	return res, nil
}
