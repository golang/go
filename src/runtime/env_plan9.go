// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

var tracebackbuf [128]byte

func gogetenv(key string) string {
	var file [128]byte
	if len(key) > len(file)-6 {
		return ""
	}

	copy(file[:], "/env/")
	copy(file[5:], key)

	fd := open(&file[0], _OREAD, 0)
	if fd < 0 {
		return ""
	}
	n := seek(fd, 0, 2)
	if n <= 0 {
		closefd(fd)
		return ""
	}

	p := make([]byte, n)

	r := pread(fd, unsafe.Pointer(&p[0]), int32(n), 0)
	closefd(fd)
	if r < 0 {
		return ""
	}

	if p[r-1] == 0 {
		r--
	}

	var s string
	sp := (*_string)(unsafe.Pointer(&s))
	sp.str = &p[0]
	sp.len = int(r)
	return s
}

var _cgo_setenv unsafe.Pointer   // pointer to C function
var _cgo_unsetenv unsafe.Pointer // pointer to C function
