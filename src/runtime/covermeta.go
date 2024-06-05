// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/coverage/rtcov"
	"unsafe"
)

// The compiler emits calls to runtime.addCovMeta
// but this code has moved to rtcov.AddMeta.
func addCovMeta(p unsafe.Pointer, dlen uint32, hash [16]byte, pkgpath string, pkgid int, cmode uint8, cgran uint8) uint32 {
	id := rtcov.AddMeta(p, dlen, hash, pkgpath, pkgid, cmode, cgran)
	if id == 0 {
		throw("runtime.addCovMeta: coverage package map collision")
	}
	return id
}
