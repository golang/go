// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func getenv(s *byte) *byte {
	val := gogetenv(gostringnocopy(s))
	if val == "" {
		return nil
	}
	// Strings found in environment are NUL-terminated.
	return &bytes(val)[0]
}

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

	var p unsafe.Pointer

	// Be sure not to allocate for $GOTRACEBACK.
	if key == "GOTRACEBACK" {
		if n >= 128 {
			return ""
		}
		p = unsafe.Pointer(&tracebackbuf[0])
	} else {
		p = gomallocgc(uintptr(n+1), nil, 0)
	}

	r := pread(fd, p, int32(n), 0)
	close(fd)
	if r < 0 {
		return ""
	}

	var s string
	sp := (*_string)(unsafe.Pointer(&s))
	sp.str = (*byte)(p)
	sp.len = int(r)
	return s
}
