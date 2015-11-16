// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/sys"
	"unsafe"
)

const (
	_AT_NULL    = 0
	_AT_RANDOM  = 25
	_AT_SYSINFO = 32
)

var _vdso uint32

func sysargs(argc int32, argv **byte) {
	// skip over argv, envv to get to auxv
	n := argc + 1
	for argv_index(argv, n) != nil {
		n++
	}
	n++
	auxv := (*[1 << 28]uint32)(add(unsafe.Pointer(argv), uintptr(n)*sys.PtrSize))

	for i := 0; auxv[i] != _AT_NULL; i += 2 {
		switch auxv[i] {
		case _AT_SYSINFO:
			_vdso = auxv[i+1]

		case _AT_RANDOM:
			startupRandomData = (*[16]byte)(unsafe.Pointer(uintptr(auxv[i+1])))[:]
		}
	}
}
